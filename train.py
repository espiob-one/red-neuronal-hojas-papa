"""Train a compact potato leaf classifier; keep the grouped test split untouched until the end."""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class LeafDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(ROOT / row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, row["class_index"]


def metrics(truth, predicted, classes):
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(cm, (truth, predicted), 1)
    tp = np.diag(cm)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)), "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()), "confusion_matrix": cm.tolist(),
        "confusion_matrix_order": "rows=true, columns=predicted; same order as classes",
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "support": int(cm[i].sum())} for i, label in enumerate(classes)},
    }


def evaluate(model, loader, device, classes):
    model.eval()
    truth, predicted, probabilities = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = logits.softmax(dim=1).cpu()
            truth.extend(labels.tolist())
            predicted.extend(probs.argmax(dim=1).tolist())
            probabilities.extend(probs.tolist())
    return metrics(truth, predicted, classes), predicted, probabilities


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--fine-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.head_epochs < 1 or args.fine_epochs < 0 or args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid training parameters")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; check GPU installation before training")
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    manifest_path = ROOT / "data/metadata/manifest.json"
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    classes = json.loads((ROOT / "data/metadata/classes.json").read_text(encoding="utf-8"))
    split_rows = {split: [row for row in rows if row["split"] == split] for split in ("train", "val", "test")}
    for key in ("group_id", "sha256", "pixel_sha256"):
        sets = {split: {row[key] for row in items} for split, items in split_rows.items()}
        if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
            raise RuntimeError(f"Data leakage in {key}")
    for split, items in split_rows.items():
        if {row["class_index"] for row in items} != set(range(len(classes))):
            raise RuntimeError(f"Missing class in {split}")

    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2
    validation_transform = weights.transforms()
    training_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(.8, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.1),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    loaders = {split: DataLoader(
        LeafDataset(items, training_transform if split == "train" else validation_transform),
        batch_size=args.batch_size, shuffle=split == "train", num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(42),
    ) for split, items in split_rows.items()}

    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.to(device)
    counts = Counter(row["class_index"] for row in split_rows["train"])
    class_weights = torch.tensor([len(split_rows["train"]) / (len(classes) * counts[i]) for i in range(len(classes))], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    run = ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args), "architecture": "mobilenet_v2", "pretrained": "IMAGENET1K_V2", "seed": 42,
        "classes": classes, "mean": MEAN, "std": STD, "resize_short_edge": 232, "center_crop": 224,
        "input": "RGB float32 NCHW, scaled to [0,1], then normalized", "output": "3 logits; softmax for scores",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "split_sizes": {split: len(items) for split, items in split_rows.items()},
        "training_class_weights": class_weights.tolist(), "selection_metric": "validation macro_f1",
        "parameters": sum(p.numel() for p in model.parameters()),
        "limitations": ["Only individual potato leaf images", "No field validation", "No non-leaf/unknown class", "Softmax scores are not calibrated disease probabilities"],
    }
    save_json(run / "config.json", config)
    history, best_f1 = [], -1
    start = time.perf_counter()
    for epoch in range(args.head_epochs + args.fine_epochs):
        if epoch == args.head_epochs:
            for parameter in model.features[-3:].parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": model.features[-3:].parameters(), "lr": 1e-4},
                {"params": model.classifier.parameters(), "lr": 3e-4},
            ], weight_decay=1e-4)
        model.train()
        # Keep pretrained batch-normalization statistics stable on this small dataset.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        loss_sum, batches = 0.0, 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            batches += 1
        val_metrics, _, _ = evaluate(model, loaders["val"], device, classes)
        record = {"epoch": epoch + 1, "phase": "head" if epoch < args.head_epochs else "fine_tune",
            "train_weighted_loss_mean_per_batch": loss_sum / batches,
            "val_accuracy": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"],
            "elapsed_seconds": round(time.perf_counter() - start, 2)}
        history.append(record)
        save_json(run / "history.json", history)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), run / "best_weights.pt")
            save_json(run / "best_validation.json", {"epoch": epoch + 1, **val_metrics})
        print(json.dumps(record), flush=True)

    model.load_state_dict(torch.load(run / "best_weights.pt", map_location=device, weights_only=True))
    test_metrics, predicted, probs = evaluate(model, loaders["test"], device, classes)
    # Multiple photos of a leaf are correlated: also score one prediction per leaf.
    leaf_predictions = {}
    for row, scores in zip(split_rows["test"], probs):
        item = leaf_predictions.setdefault(row["group_id"], {"true": row["class_index"], "scores": []})
        if item["true"] != row["class_index"]:
            raise RuntimeError("A leaf group has conflicting labels")
        item["scores"].append(scores)
    leaf_truth = [item["true"] for item in leaf_predictions.values()]
    leaf_predicted = [int(np.mean(item["scores"], axis=0).argmax()) for item in leaf_predictions.values()]
    leaf_metrics = metrics(leaf_truth, leaf_predicted, classes)
    save_json(run / "test_metrics.json", {"classes": classes, **test_metrics,
        "leaf_level": {"aggregation": "mean softmax scores per leaf group", **leaf_metrics}})
    save_json(run / "test_predictions.json", [
        {"path": row["path"], "leaf_group": row["group_id"], "true_class": row["label"],
         "predicted_class": classes[pred], "scores": scores}
        for row, pred, scores in zip(split_rows["test"], predicted, probs)
    ])
    print("TEST:", json.dumps(test_metrics), flush=True)
    print("TEST BY LEAF:", json.dumps(leaf_metrics), flush=True)
    print("RUN:", run, flush=True)


if __name__ == "__main__":
    main()
