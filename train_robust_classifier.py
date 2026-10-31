"""Train a background/geometry robustness candidate without camera-image tuning.

Only the existing leaf-grouped train and validation partitions are consumed.
The clean validation transform remains the original transform. A separate,
deterministic synthetic challenge uses segmented validation leaves; it is not
evidence of field or printed-photo accuracy. The test partition is never run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CROP_MARGIN = .12
SEED = 20261022


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def seed_for(path):
    return int.from_bytes(hashlib.sha256((str(SEED) + path).encode()).digest()[:8], "little")


def foreground(image, original=None):
    """Remove only near-black regions connected to the image boundary.

    Dark disease tissue enclosed inside the leaf stays opaque. Segmented JPEGs
    have a black exterior, so no color/greenness assumption is needed.
    """
    pixels = np.asarray(image.convert("RGB"))
    # Copy explicitly: fromarray can be read-only and PIL floodfill's PixelAccess
    # writes would otherwise fail to update the actual image storage.
    mask = Image.fromarray((pixels.max(axis=2) > 20).astype(np.uint8)).copy()
    access = mask.load()
    w, h = mask.size
    boundary = [(x, y) for x in range(w) for y in (0, h - 1)]
    boundary += [(x, y) for y in range(h) for x in (0, w - 1)]
    for xy in boundary:
        if access[xy] == 0:
            ImageDraw.floodfill(mask, xy, 2, thresh=0)
    alpha = Image.fromarray(np.uint8(np.asarray(mask) != 2) * 255)
    fraction = float((np.asarray(alpha) > 0).mean())
    if not .08 <= fraction <= .95:
        raise ValueError(f"Implausible segmented leaf alpha fraction: {fraction}")
    if original is not None and original.size != image.size:
        raise ValueError("Segmented mask and original RGB have different geometry")
    # The upstream segmented JPEG changes brightness; use only its geometry,
    # preserving tissue texture and RGB from the original source photograph.
    result = (original if original is not None else image).convert("RGBA")
    result.putalpha(alpha)
    box = alpha.getbbox()
    if box is None:
        raise ValueError("Segmented image has no foreground")
    return result.crop(box)


def verify_generator():
    """Check alpha/color invariants before accepting augmented data."""
    segmented = Image.new("RGB", (32, 32), (0, 0, 0))
    draw = ImageDraw.Draw(segmented)
    draw.rectangle((8, 8, 23, 23), fill=(30, 140, 35))
    draw.rectangle((14, 14, 17, 17), fill=(0, 0, 0))
    original_pixels = np.random.default_rng(21).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    original = Image.fromarray(original_pixels)
    result = foreground(segmented, original)
    assert result.size == (16, 16), "Exterior black was not removed"
    assert result.getpixel((8, 8))[3] == 255, "Dark interior disease tissue became transparent"
    np.testing.assert_array_equal(np.asarray(result)[:, :, :3], original_pixels[8:24, 8:24])
    # A leaf with a notch verifies that outside background inside the bounding
    # rectangle is transparent, rather than simply cropping a solid rectangle.
    draw.rectangle((8, 8, 11, 15), fill=(0, 0, 0))
    result = foreground(segmented, original)
    assert result.getpixel((0, 0))[3] == 0, "Connected exterior black stayed opaque"
    assert result.getpixel((8, 8))[3] == 255
    classes = json.loads((ROOT / "data/metadata/classes.json").read_text())
    rows = json.loads((ROOT / "data/metadata/manifest.json").read_text())
    checked = []
    for label in classes:
        candidates = sorted((r for r in rows if r["split"] == "val" and r["label"] == label), key=lambda r: r["path"])[:2]
        for row in candidates:
            with Image.open(ROOT / row["path"]) as image:
                rgb = image.convert("RGB")
            path = ROOT / "data/segmented" / label / (Path(row["path"]).stem + "_final_masked.jpg")
            with Image.open(path) as image:
                leaf = foreground(image.convert("RGB"), rgb)
            alpha = np.asarray(leaf.getchannel("A"))
            assert float((alpha > 0).mean()) < .95, "Real sample incorrectly became an opaque rectangle"
            assert leaf.size != rgb.size, "Real sample retained the full original bounding rectangle"
            rng = random.Random(seed_for(row["path"]))
            generated = photometric(compose(leaf, rng), rng, True)
            rng2 = random.Random(seed_for(row["path"]))
            repeated = photometric(compose(leaf, rng2), rng2, True)
            np.testing.assert_array_equal(np.asarray(generated), np.asarray(repeated))
            checked.append({"path": row["path"], "leaf_size": list(leaf.size),
                            "alpha_nonzero_fraction_in_leaf_bbox": float((alpha > 0).mean()),
                            "deterministic_challenge": True})
    return {"synthetic_alpha_exterior_black": True, "synthetic_alpha_interior_black": True,
            "original_rgb_exact": True, "real_samples": checked, "passed": True}


def background(rng, size):
    # Includes neutral/white paper, grey studio, green vegetation and earth;
    # low-frequency texture avoids a single synthetic background shortcut.
    palettes = [(240, 239, 231), (181, 181, 181), (105, 102, 94),
                (66, 93, 47), (107, 83, 54), (154, 172, 135),
                (211, 220, 230), (75, 79, 86)]
    base = np.array(rng.choice(palettes), dtype=np.float32)
    if rng.random() < .25:
        base[:] = rng.uniform(70, 248)
    noise_rng = np.random.default_rng(rng.getrandbits(63))
    noise = noise_rng.normal(0, 1, (12, 12, 1))
    noise = noise + noise_rng.normal(0, .25, (12, 12, 3))
    array = np.uint8(np.clip(base + noise * rng.uniform(3, 16), 0, 255))
    return Image.fromarray(array).resize(size, Image.Resampling.BICUBIC)


def photometric(image, rng, strong):
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(.76, 1.24) if strong else rng.uniform(.90, 1.10))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(.78, 1.20) if strong else rng.uniform(.92, 1.10))
    image = ImageEnhance.Color(image).enhance(rng.uniform(.62, 1.20) if strong else rng.uniform(.90, 1.10))
    # Mild white-balance change simulates indoor illumination/printing without
    # replacing disease tissue or converting labels by a color heuristic.
    if strong:
        pixels = np.asarray(image, dtype=np.float32)
        gains = np.array([rng.uniform(.8, 1.2) for _ in range(3)])
        gamma = rng.uniform(.8, 1.2)
        image = Image.fromarray(np.uint8(np.clip(255 * ((pixels / 255) ** gamma) * gains, 0, 255)))
    if rng.random() < (.45 if strong else .15):
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(.15, .85)))
    if rng.random() < (.6 if strong else .15):
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=rng.randint(63, 96))
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
    return image


def compose(leaf, rng):
    """Generate a detected crop with 12% margin and varied capture geometry."""
    leaf = leaf.copy()
    if rng.random() < .5:
        leaf = ImageOps.mirror(leaf)
    if rng.random() < .5:
        leaf = ImageOps.flip(leaf)
    # Place on a transparent square before perspective distortion so that the
    # whole leaf survives the operation. No crop removes lesion-bearing areas.
    side = max(leaf.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.alpha_composite(leaf, ((side - leaf.width) // 2, (side - leaf.height) // 2))
    j = rng.uniform(.01, .10) * side
    start = [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]]
    end = [[rng.uniform(0, j), rng.uniform(0, j)],
           [side - 1 - rng.uniform(0, j), rng.uniform(0, j)],
           [side - 1 - rng.uniform(0, j), side - 1 - rng.uniform(0, j)],
           [rng.uniform(0, j), side - 1 - rng.uniform(0, j)]]
    square = TF.perspective(square, start, end, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
    square = square.rotate(rng.uniform(-32, 32), resample=Image.Resampling.BICUBIC, expand=True)
    box = square.getchannel("A").getbbox()
    if box is None:
        raise RuntimeError("Empty transformed leaf")
    leaf = square.crop(box)
    # Frame aspect and scale vary before the same square pad used at runtime.
    scale = rng.uniform(.72, 1.0)
    target_max = 224 * scale
    ratio = target_max / max(leaf.size)
    leaf = leaf.resize((max(1, round(leaf.width * ratio)), max(1, round(leaf.height * ratio))), Image.Resampling.BILINEAR)
    width = max(8, round(leaf.width * (1 + 2 * CROP_MARGIN)))
    height = max(8, round(leaf.height * (1 + 2 * CROP_MARGIN)))
    canvas = background(rng, (width, height))
    jitter_x = rng.randint(-max(1, round(width * .025)), max(1, round(width * .025)))
    jitter_y = rng.randint(-max(1, round(height * .025)), max(1, round(height * .025)))
    canvas.paste(leaf, ((width - leaf.width) // 2 + jitter_x, (height - leaf.height) // 2 + jitter_y), leaf)
    return ImageOps.pad(canvas, (max(width, height),) * 2, method=Image.Resampling.BILINEAR, color=(128, 128, 128))


class LeafDataset(Dataset):
    def __init__(self, rows, mode):
        self.rows, self.mode = rows, mode
        self.challenge_tensors = None
        self.originals, self.leaves = [], []
        self.transform = transforms.Compose([transforms.Resize(232), transforms.CenterCrop(224),
                                             transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        for row in rows:
            with Image.open(ROOT / row["path"]) as image:
                self.originals.append(image.convert("RGB"))
            if mode != "clean":
                stem = Path(row["path"]).stem
                candidates = [ROOT / "data/segmented" / row["label"] / (stem + "_final_masked.jpg"),
                              ROOT / "data/segmented" / row["label"] / Path(row["path"]).name]
                path = next((p for p in candidates if p.exists()), None)
                if path is None:
                    raise FileNotFoundError(f"Missing segmented leaf {row['path']}")
                with Image.open(path) as image:
                    self.leaves.append(foreground(image.convert("RGB"), self.originals[-1]))
        if mode == "challenge":
            self.challenge_tensors = [self[i][0] for i in range(len(rows))]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.challenge_tensors is not None:
            return self.challenge_tensors[index], self.rows[index]["class_index"]
        image = self.originals[index].copy()
        if self.mode != "clean":
            rng = random.Random(seed_for(self.rows[index]["path"])) if self.mode == "challenge" else random
            synthetic = self.mode == "challenge" or rng.random() >= .40
            if synthetic:
                image = compose(self.leaves[index], rng)
            else:
                if rng.random() < .5:
                    image = ImageOps.mirror(image)
                if rng.random() < .5:
                    image = ImageOps.flip(image)
                image = image.rotate(rng.uniform(-10, 10), resample=Image.Resampling.BILINEAR)
            image = photometric(image, rng, strong=synthetic)
        return self.transform(image), self.rows[index]["class_index"]


def metrics(rows, probabilities, classes):
    truth = np.array([r["class_index"] for r in rows])
    predicted = probabilities.argmax(axis=1)
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(cm, (truth, predicted), 1)
    tp = cm.diagonal()
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    result = {"accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1.mean()),
              "confusion_matrix": cm.tolist(), "confusion_order": "rows=true, columns=predicted",
              "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]),
                                "f1": float(f1[i]), "support": int(cm[i].sum())} for i, c in enumerate(classes)}}
    return result


def summarize(rows, probabilities, classes):
    result = metrics(rows, probabilities, classes)
    groups = defaultdict(list)
    for row, scores in zip(rows, probabilities):
        groups[row["group_id"]].append((row, scores))
    group_rows, group_scores = [], []
    for items in groups.values():
        if len({r["class_index"] for r, _ in items}) != 1:
            raise RuntimeError("Conflicting group labels")
        group_rows.append(items[0][0])
        group_scores.append(np.mean([scores for _, scores in items], axis=0))
    result["leaf_group_level"] = metrics(group_rows, np.array(group_scores), classes)
    return result


def evaluate(model, loader, device, classes):
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for images, _ in loader:
            # Full float32 evaluation avoids selecting on mixed precision noise.
            probabilities.append(model(images.to(device)).softmax(dim=1).cpu().numpy())
    probs = np.concatenate(probabilities)
    return summarize(loader.dataset.rows, probs, classes), probs


def create_model(classes, weights):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    return model


def export_onnx(model, output, config, loaders):
    import onnx
    import onnxruntime as ort
    output.mkdir(parents=True, exist_ok=True)
    path = output / "model.onnx"
    model = model.eval().cpu()
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), str(path), input_names=["image"],
                      output_names=["logits"], opset_version=17, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    comparisons = []
    with torch.inference_mode():
        for mode in ("clean", "challenge"):
            rows = loaders[mode].dataset.rows
            for label in config["classes"]:
                indices = [i for i, r in enumerate(rows) if r["label"] == label][:4]
                for i in indices:
                    tensor = loaders[mode].dataset[i][0].unsqueeze(0)
                    reference = model(tensor).numpy()
                    result = session.run(None, {"image": tensor.numpy()})[0]
                    comparisons.append({"mode": mode, "path": rows[i]["path"],
                                        "max_absolute_logit_difference": float(np.abs(reference - result).max()),
                                        "argmax_match": bool(reference.argmax() == result.argmax())})
    check = {"max_absolute_logit_difference": max(x["max_absolute_logit_difference"] for x in comparisons),
             "all_argmax_match": all(x["argmax_match"] for x in comparisons), "comparisons": comparisons}
    check["passed"] = check["max_absolute_logit_difference"] <= 1e-4 and check["all_argmax_match"]
    save_json(output / "export_validation.json", check)
    if not check["passed"]:
        raise RuntimeError("ONNX numerical equivalence failed")
    save_json(output / "config.json", config)
    return check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    generator_check = verify_generator()
    print("GENERATOR_CHECK " + json.dumps(generator_check), flush=True)
    if args.check_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.set_num_threads(4)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    manifest_path = ROOT / "data/metadata/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    classes = json.loads((ROOT / "data/metadata/classes.json").read_text())
    partitions = {s: [r for r in manifest if r["split"] == s] for s in ("train", "val", "test")}
    for key in ("group_id", "sha256", "pixel_sha256"):
        sets = {s: {r[key] for r in rs} for s, rs in partitions.items()}
        if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
            raise RuntimeError(f"Split leakage: {key}")
    rows_train, rows_val = partitions["train"], partitions["val"]
    # Test rows participate only in the overlap audit above; their image files,
    # predictions and performance are never consumed for candidate selection.
    run = ROOT / "runs" / ("robust_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "generator_check.json", generator_check)
    source_weights = ROOT / "runs/20260911_213926/best_weights.pt"
    datasets = {"train": LeafDataset(rows_train, "train"),
                "clean": LeafDataset(rows_val, "clean"), "challenge": LeafDataset(rows_val, "challenge")}
    counts = Counter(r["class_index"] for r in rows_train)
    sampler = WeightedRandomSampler([1 / counts[r["class_index"]] for r in rows_train],
                                    len(rows_train), replacement=True, generator=torch.Generator().manual_seed(SEED))
    loaders = {mode: DataLoader(dataset, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, sampler=sampler if mode == "train" else None)
               for mode, dataset in datasets.items()}
    config = {**vars(args), "architecture": "mobilenet_v2", "classes": classes, "seed": SEED,
              "mean": MEAN, "std": STD, "resize_short_edge": 232, "center_crop": 224,
              "crop_margin": CROP_MARGIN, "square_pad": True, "square_pad_color": [128, 128, 128],
              "input": "RGB float32 NCHW normalized ImageNet; crop box expands 12%, pad gray128 square, resize232 center224",
              "output": "3 logits; softmax scores are not calibrated disease probabilities",
              "source_weights": str(source_weights), "source_weights_sha256": hashlib.sha256(source_weights.read_bytes()).hexdigest(),
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "split_sizes": {"train": len(rows_train), "val": len(rows_val)},
              "balance": "inverse-frequency WeightedRandomSampler only; unweighted cross entropy",
              "training": "40% original RGB with mild augment; 60% original RGB tissue with boundary-floodfilled segmented alpha mask, varied textured backgrounds, geometry/color/blur/JPEG; channel gains0.8-1.2 gamma0.8-1.2 same distribution for all labels",
              "clean_validation": "Original RGB resize232 center224, same original baseline preprocessing",
              "challenge_validation": "Per-image SHA256-seeded synthetic composition; fixed seed; runtime crop margin and square padding",
              "selection": "mean clean/challenge macroF1; clean accuracy no more than 0.02 below baseline; each disease clean recall no more than 0.03 below baseline; challenge macroF1 must improve",
              "test_used": False, "user_camera_images_used": False,
              "limitations": ["Synthetic validation only; no field or printed-camera holdout validation", "Small healthy class (104 training images)",
                              "Potato only; no unknown/non-leaf class", "Dataset photos within leaf groups are correlated"],
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    save_json(run / "config.json", config)
    print(json.dumps({"run": str(run), "gpu": config["gpu"], "split_sizes": config["split_sizes"]}), flush=True)
    model = create_model(classes, source_weights).to(device)
    baseline, _ = {}, None
    for mode in ("clean", "challenge"):
        baseline[mode], _ = evaluate(model, loaders[mode], device, classes)
    save_json(run / "baseline_validation.json", baseline)
    print("BASELINE " + json.dumps(baseline), flush=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.features[-5:].parameters():
        parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW([{"params": model.features[-5:].parameters(), "lr": 1e-4},
                                   {"params": model.classifier.parameters(), "lr": 3e-4}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=2e-5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")
    history, best_score, best = [], -1., None
    baseline_score = (baseline["clean"]["macro_f1"] + baseline["challenge"]["macro_f1"]) / 2
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Freeze all BatchNorm running statistics, including fine-tuned blocks.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        total_loss, batches = 0., 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            batches += 1
        results = {}
        for mode in ("clean", "challenge"):
            results[mode], _ = evaluate(model, loaders[mode], device, classes)
        score = (results["clean"]["macro_f1"] + results["challenge"]["macro_f1"]) / 2
        accuracy_guard = results["clean"]["accuracy"] >= baseline["clean"]["accuracy"] - .02
        disease_guard = all(results["clean"]["per_class"][c]["recall"] >=
                            baseline["clean"]["per_class"][c]["recall"] - .03
                            for c in classes if not c.endswith("healthy"))
        guard = accuracy_guard and disease_guard
        record = {"epoch": epoch, "train_loss": total_loss / batches,
                  "selection_score": score, "clean_accuracy_guard": accuracy_guard,
                  "clean_disease_recall_guard": disease_guard, "all_selection_guards": guard, **results,
                  "elapsed_seconds": round(time.perf_counter() - start, 2)}
        history.append(record)
        save_json(run / "history.json", history)
        if guard and score > best_score:
            best_score, best = score, record
            torch.save(model.state_dict(), run / "best_weights.pt")
            save_json(run / "best_validation.json", record)
        scheduler.step()
        print("EPOCH " + json.dumps(record), flush=True)
    passed = best is not None and best_score > baseline_score and best["challenge"]["macro_f1"] > baseline["challenge"]["macro_f1"]
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "run": str(run),
              "baseline": baseline, "selected": best, "baseline_selection_score": baseline_score,
              "selected_score": best_score, "passed_promotion_guard": passed, "deployed": False,
              "test_used": False, "user_camera_images_used": False, "history": history,
              "limitations": config["limitations"]}
    if passed:
        model.load_state_dict(torch.load(run / "best_weights.pt", weights_only=True, map_location=device))
        output = ROOT / "models/potato-robust-candidate"
        report["onnx_export"] = export_onnx(model, output, config, loaders)
        report["output"] = str(output)
    save_json(run / "robust_validation.json", report)
    save_json(ROOT / "robust_validation.json", report)
    print("RESULT " + json.dumps({"run": str(run), "passed_promotion_guard": passed,
                                  "best_epoch": None if best is None else best["epoch"],
                                  "baseline_score": baseline_score, "best_score": best_score,
                                  "output": report.get("output"), "deployed": False}), flush=True)


if __name__ == "__main__":
    main()
