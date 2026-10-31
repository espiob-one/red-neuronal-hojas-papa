"""Deterministic diagnostic validation and multi-object synthetic video replay.

This suite never trains or deploys, consumes only the existing validation split,
and keeps model/source snapshots. Existing validation data have already informed
candidate selection: these results are stress diagnostics, not fresh test scores.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms
from torchvision.transforms import functional as TF

from fast_leaf_detector import FastLeafDetector
from leaf_tracks import LeafTracker, intersection_over_union
from rover_motion import capture_ready, classifier_input, leaf_crop
from train_robust_classifier import foreground


ROOT = Path(__file__).resolve().parent
SEED = 202610220741
CONDITIONS = ("clean", "backgrounds", "perspective_scale", "motion_blur", "gamma_rgb", "jpeg_occlusion")
PALETTES = [(128, 128, 128), (225, 222, 208), (51, 151, 166), (54, 84, 42), (115, 78, 45)]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fixed_seed(key):
    return int.from_bytes(sha256(f"{SEED}:{key}".encode()).digest()[:8], "little")


def texture(size, key):
    rng = np.random.default_rng(fixed_seed(key))
    base = np.asarray(PALETTES[fixed_seed(key) % len(PALETTES)], dtype=np.float32)
    noise = rng.normal(0, 6, (13, 13, 3))
    return Image.fromarray(np.uint8(np.clip(base + noise, 0, 255))).resize(size, Image.Resampling.BICUBIC)


def leaf_asset(row):
    rgb_path = ROOT / row["path"]
    segmented_path = ROOT / "data/segmented" / row["label"] / (rgb_path.stem + "_final_masked.jpg")
    with Image.open(rgb_path) as source:
        rgb = source.convert("RGB")
    with Image.open(segmented_path) as source:
        asset = foreground(source.convert("RGB"), rgb)
    return rgb, asset, {"original_sha256": digest(rgb_path), "segmented_sha256": digest(segmented_path),
                        "asset_size": list(asset.size),
                        "alpha_nonzero_fraction": float((np.asarray(asset.getchannel("A")) > 0).mean())}


def motion_blur(image, length, direction):
    """Convolve with a normalized linear exposure kernel, using edge padding."""
    if length % 2 != 1 or length < 1 or direction not in ("h", "v", "d", "a"):
        raise ValueError("Odd kernel length and known linear direction required")
    radius = length // 2
    pixels = np.asarray(image, dtype=np.float32)
    padded = np.pad(pixels, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    result = np.zeros_like(pixels)
    for delta in range(-radius, radius + 1):
        dy = 0 if direction == "h" else delta
        dx = 0 if direction == "v" else (-delta if direction == "a" else delta)
        result += padded[radius + dy:radius + dy + image.height,
                         radius + dx:radius + dx + image.width]
    return Image.fromarray(np.uint8(np.clip(np.rint(result / length), 0, 255)))


def compose_condition(rgb, asset, row, condition):
    key = row["path"]
    rng = random.Random(fixed_seed(key))
    if condition == "clean":
        return rgb.copy(), {"original": True}
    leaf = asset.copy()
    details = {"background_palette": PALETTES[fixed_seed(key) % len(PALETTES)]}
    if condition == "perspective_scale":
        side = max(leaf.size) + 20
        square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        square.alpha_composite(leaf, ((side - leaf.width) // 2, (side - leaf.height) // 2))
        corners = [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]]
        target = [[round(.10 * side), round(.04 * side)], [round(.94 * side), 0],
                  [side - 1, round(.93 * side)], [0, round(.85 * side)]]
        square = TF.perspective(square, corners, target,
                                interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
        angle = -18 if rng.getrandbits(1) else 18
        square = square.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        leaf = square.crop(square.getchannel("A").getbbox())
        details.update(perspective_endpoints=target, rotation_degrees=angle, max_leaf_side=166)
    target_max = 166 if condition == "perspective_scale" else 206
    ratio = target_max / max(leaf.size)
    leaf = leaf.resize((max(1, round(leaf.width * ratio)), max(1, round(leaf.height * ratio))),
                       Image.Resampling.BILINEAR)
    canvas = texture((256, 256), key)
    x = (256 - leaf.width) // 2 + (12 if condition == "perspective_scale" else 0)
    y = (256 - leaf.height) // 2 + (-9 if condition == "perspective_scale" else 0)
    canvas.paste(leaf, (x, y), leaf)
    if condition == "motion_blur":
        direction = ("h", "v", "d", "a")[fixed_seed(key) % 4]
        canvas = motion_blur(canvas, 9, direction)
        details.update(linear_kernel_length=9, linear_kernel_direction=direction)
    elif condition == "gamma_rgb":
        warm = bool(fixed_seed(key) % 2)
        gains, gamma = ([1.18, .90, .82], 1.20) if warm else ([.82, .96, 1.18], .78)
        pixels = np.asarray(canvas, dtype=np.float32) / 255
        canvas = Image.fromarray(np.uint8(np.clip(255 * (pixels ** gamma) * gains, 0, 255)))
        details.update(rgb_gains=gains, gamma=gamma)
    elif condition == "jpeg_occlusion":
        width, height = max(1, round(leaf.width * .17)), max(1, round(leaf.height * .28))
        ox, oy = x + round(leaf.width * .72) - width // 2, y + round(leaf.height * .55) - height // 2
        alpha = np.asarray(leaf.getchannel("A")) > 0
        removed = alpha[max(0, oy - y):min(leaf.height, oy - y + height),
                        max(0, ox - x):min(leaf.width, ox - x + width)].sum()
        ImageDraw.Draw(canvas).rectangle((ox, oy, ox + width - 1, oy + height - 1), fill=(118, 112, 98))
        buffer = BytesIO()
        canvas.save(buffer, format="JPEG", quality=42, subsampling=2)
        buffer.seek(0)
        canvas = Image.open(buffer).convert("RGB")
        details.update(jpeg_quality=42, occlusion_box=[ox, oy, ox + width, oy + height],
                       alpha_fraction_occluded=float(removed / max(1, alpha.sum())))
    return canvas, details


class Classifier:
    def __init__(self, model_path, config_path, threads):
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.clean_transform = transforms.Compose([
            transforms.Resize(self.config["resize_short_edge"]), transforms.CenterCrop(self.config["center_crop"]),
            transforms.ToTensor(), transforms.Normalize(self.config["mean"], self.config["std"]),
        ])

    def predict(self, image, clean=False):
        tensor = (self.clean_transform(image).unsqueeze(0).numpy() if clean
                  else classifier_input(image, self.config))
        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        scores = np.exp(logits - logits.max())
        return scores / scores.sum()


def classification_metrics(rows, scores, classes):
    truth = np.asarray([row["class_index"] for row in rows])
    scores = np.asarray(scores)
    predicted = scores.argmax(axis=1)
    confidence = scores.max(axis=1)
    ordered = np.sort(scores, axis=1)
    accepted = (confidence >= .7) & ((ordered[:, -1] - ordered[:, -2]) >= .2)
    wrong = predicted != truth
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(matrix, (truth, predicted), 1)
    tp = matrix.diagonal()
    precision = tp / np.maximum(matrix.sum(axis=0), 1)
    recall = tp / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    most_wrong = np.flatnonzero(wrong)[np.argsort(-confidence[wrong])][:20]
    bins = []
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        selected = (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
        bins.append({"low": float(low), "high": float(high), "count": int(selected.sum()),
                     "accuracy": float((~wrong)[selected].mean()) if selected.any() else None})
    return {"accuracy": float((~wrong).mean()), "macro_f1": float(f1.mean()),
            "confusion_matrix": matrix.tolist(), "confusion_order": "rows=true, columns=predicted",
            "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                                  "f1": float(f1[i]), "support": int(matrix[i].sum())}
                          for i, label in enumerate(classes)},
            "errors": int(wrong.sum()), "high_confidence_errors": int((wrong & accepted).sum()),
            "high_confidence_error_fraction_all": float((wrong & accepted).mean()),
            "accepted_fraction": float(accepted.mean()),
            "accuracy_among_accepted": float((~wrong)[accepted].mean()) if accepted.any() else None,
            "confidence_bins": bins,
            "most_confident_errors": [{"path": rows[i]["path"], "group_id": rows[i]["group_id"],
                                       "truth": classes[int(truth[i])], "predicted": classes[int(predicted[i])],
                                       "confidence": float(confidence[i])} for i in most_wrong]}


def grouped_metrics(rows, scores, classes):
    groups = defaultdict(list)
    for row, value in zip(rows, scores):
        groups[row["group_id"]].append((row, value))
    group_rows, group_scores, predictions = [], [], []
    for group_id, items in sorted(groups.items()):
        if len({item[0]["class_index"] for item in items}) != 1:
            raise RuntimeError("Group with conflicting labels")
        row = items[0][0]
        score = np.mean([item[1] for item in items], axis=0)
        group_rows.append(row)
        group_scores.append(score)
        predictions.append({"group_id": group_id, "image_count": len(items), "truth": row["label"],
                            "predicted": classes[int(score.argmax())], "scores": [float(x) for x in score]})
    return {"images": classification_metrics(rows, scores, classes),
            "groups": classification_metrics(group_rows, group_scores, classes), "group_predictions": predictions}


def read_negatives(path):
    if not path.exists():
        return [], {"status": "skipped", "reason": "Public annotated non-leaf manifest not available", "path": str(path)}
    parsed = json.loads(path.read_text(encoding="utf-8"))
    rows = parsed if isinstance(parsed, list) else parsed.get("rows", parsed.get("images", []))
    accepted = [row for row in rows if row.get("split") == "val" and row.get("is_leaf") is False
                and float(row.get("vegetation_fraction", 1)) == 0 and float(row.get("unknown_fraction", 1)) <= .02
                and float(row.get("region_purity", 0)) >= .90 and (ROOT / row["path"]).exists()]
    accepted.sort(key=lambda row: (row.get("semantic_role") != "person_region", row["path"]))
    return accepted, {"status": "available", "path": str(path), "sha256": digest(path),
                      "eligible_validation_crops": len(accepted),
                      "filters": "val, is_leaf=false, annotated vegetation=0, unknown<=2%, purity>=90%; no user photos",
                      "limitation": "Panoptic rule-filtered crops can contain unannotated incidental objects; not an absolute no-leaf guarantee."}


def evaluate_negatives(rows, detector, classifiers, classes, output):
    records = []
    for row in rows[:60]:
        with Image.open(ROOT / row["path"]) as source:
            image = source.convert("RGB")
        detections = detector.detect(image)
        results = []
        for detection in detections:
            crop = leaf_crop(image, detection["box"])
            ready = capture_ready(image, detection["box"], crop)
            scores = {name: model.predict(crop) for name, model in classifiers.items()}
            results.append({**detection, "capture_ready": ready,
                            "classifiers": {name: {"top_class": classes[int(prob.argmax())],
                                                   "confidence": float(prob.max()), "scores": prob.tolist()}
                                            for name, prob in scores.items()}})
        records.append({"path": row["path"], "group_id": row.get("group_id"),
                        "semantic_role": row.get("semantic_role"), "detections": results})
    write_json(output / "negative_proposals.json", records)
    confident_ready = {name: 0 for name in classifiers}
    for record in records:
        for proposal in record["detections"]:
            for name, result in proposal["classifiers"].items():
                ordered = sorted(result["scores"])
                confident_ready[name] += int(proposal["capture_ready"] and ordered[-1] >= .7
                                             and ordered[-1] - ordered[-2] >= .2)
    return {"images": len(records), "scenes": len({row["group_id"] for row in records}),
            "images_with_false_leaf_proposal": sum(bool(row["detections"]) for row in records),
            "false_proposals": sum(len(row["detections"]) for row in records),
            "capture_ready_false_proposals": sum(d["capture_ready"] for row in records for d in row["detections"]),
            "capture_ready_confident_nonleaf_diagnoses_by_model": confident_ready,
            "limitation": "Tests detector plus three-class classifiers before a learned leaf verifier; not the final gated pipeline."}


def render_video_frame(frame_index, sources, negative_background=None):
    image = texture((640, 360), "replay-background")
    if negative_background is not None and (frame_index < 12 or frame_index >= 110 or 55 <= frame_index < 70):
        image = ImageOps.fit(negative_background, (640, 360), method=Image.Resampling.BILINEAR)
    sprites = []
    schedules = [(12, 91, 180), (30, 103, 160), (74, 109, 140)]
    for index, (start, end, size) in enumerate(schedules):
        if not start <= frame_index <= end:
            continue
        phase = (frame_index - start) / (end - start)
        asset = sources[index][1]
        ratio = size / max(asset.size)
        leaf = asset.resize((round(asset.width * ratio), round(asset.height * ratio)), Image.Resampling.BILINEAR)
        if index == 0:
            x, y = round(-180 + phase * 900), round(70 + 14 * np.sin(phase * np.pi))
        elif index == 1:
            x, y = round(650 - phase * 850), 133
        else:
            x, y = 452, round(360 - phase * 560)
        alpha_canvas = Image.new("L", (640, 360), 0)
        alpha_canvas.paste(leaf.getchannel("A"), (x, y))
        mask = np.asarray(alpha_canvas) > 30
        total = int((np.asarray(leaf.getchannel("A")) > 30).sum())
        image.paste(leaf, (x, y), leaf)
        sprites.append({"object_id": f"leaf-{index}", "row": sources[index][0],
                        "mask": mask, "original_alpha_pixels": total})
    ground_truth = []
    covered = np.zeros((360, 640), dtype=bool)
    for sprite in reversed(sprites):
        visible = sprite["mask"] & ~covered
        covered |= sprite["mask"]
        yy, xx = np.where(visible)
        if len(xx):
            ground_truth.append({"object_id": sprite["object_id"], "label": sprite["row"]["label"],
                                 "box": [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1],
                                 "visible_alpha_fraction": float(len(xx) / sprite["original_alpha_pixels"]),
                                 "eligible": len(xx) >= 400 and len(xx) / sprite["original_alpha_pixels"] >= .4})
    return image, ground_truth


def replay(rows, assets, detector, classifiers, classes, output, negative_rows, ffmpeg):
    sources = []
    # Select first paths per class before inference; no search for detectable leaves.
    for label in (classes[0], classes[2], classes[1]):
        row = next(row for row in rows if row["label"] == label)
        sources.append((row, assets[row["path"]]))
    background = None
    background_path = None
    if negative_rows:
        selected = negative_rows[0]
        background_path = selected["path"]
        with Image.open(ROOT / background_path) as image:
            background = image.convert("RGB")
    tracker = LeafTracker()
    observations = defaultdict(list)
    records, classification_records = [], []
    totals = Counter()
    video_path = output / "replay_120frames.mp4"
    encoder = None
    encoder_error = None
    if ffmpeg:
        encoder = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                    "-s", "640x360", "-r", "8", "-i", "pipe:0", "-an", "-c:v", "libx264",
                                    "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p", str(video_path)],
                                   stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for index in range(120):
            timestamp = 1000.0 + index / 8
            image, truth = render_video_frame(index, sources, background)
            started = time.perf_counter()
            detections = detector.detect(image)
            detector_ms = (time.perf_counter() - started) * 1000
            eligible = [row for row in truth if row["eligible"]]
            pairs = sorted(((-intersection_over_union(d["box"], g["box"]), di, gi)
                            for di, d in enumerate(detections) for gi, g in enumerate(truth)))
            assignment, used_truth = {}, set()
            for neg_iou, di, gi in pairs:
                if -neg_iou < .3:
                    break
                if di not in assignment and gi not in used_truth:
                    assignment[di] = gi
                    used_truth.add(gi)
            totals["eligible_truth_observations"] += len(eligible)
            totals["missed_eligible_truth"] += sum(i not in used_truth for i, g in enumerate(truth) if g["eligible"])
            totals["false_leaf_proposals"] += len(detections) - len(assignment)
            if not truth:
                totals["blank_or_negative_frames"] += 1
                totals["blank_or_negative_frames_with_proposals"] += bool(detections)
            for detection in detections:
                crop = leaf_crop(image, detection["box"])
                detection["ready"] = capture_ready(image, detection["box"], crop)
            tracks = tracker.update(detections, timestamp)
            frame_tracks = []
            for track in tracks:
                di = track["detection_index"]
                target = truth[assignment[di]] if di in assignment else None
                if target:
                    observations[target["object_id"]].append({"frame": index, "track_id": track["track_id"]})
                if track["ready"]:
                    token = tracker.begin_classification(track["track_id"], track["last_seen"])
                    if token is not None:
                        crop = leaf_crop(image, track["box"])
                        predicted = {name: model.predict(crop) for name, model in classifiers.items()}
                        result = {"source_object_id": target["object_id"] if target else None,
                                  "classes": {name: classes[int(prob.argmax())] for name, prob in predicted.items()}}
                        accepted = tracker.finish_classification(token, result, timestamp + .0001)
                        classification_records.append({"frame": index, "track_id": track["track_id"],
                                                       "truth": target, "accepted": accepted,
                                                       "scores": {name: value.tolist() for name, value in predicted.items()}})
                result = next((t["result"] for t in tracker.snapshot()["tracks"] if t["track_id"] == track["track_id"]), None)
                wrong_source = bool(result and target and result["source_object_id"] != target["object_id"])
                totals["retained_result_on_different_object_observations"] += wrong_source
                frame_tracks.append({"track_id": track["track_id"], "box": track["box"],
                                     "matched_object": target["object_id"] if target else None,
                                     "quality_ok": track["quality_ok"], "result": result,
                                     "retained_result_wrong_source": wrong_source})
            records.append({"frame": index, "simulated_timestamp": timestamp, "ground_truth": truth,
                            "detections": detections, "tracks": frame_tracks, "detector_ms_pc": detector_ms})
            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)
            for g in truth:
                draw.rectangle(g["box"], outline=(50, 220, 70), width=1)
            for t in frame_tracks:
                draw.rectangle(t["box"], outline=(255, 100, 40), width=2)
                draw.text((max(0, t["box"][0]), max(0, t["box"][1] - 14)), f"ID {t['track_id']}", fill="white", stroke_width=1, stroke_fill="black")
            draw.text((8, 8), f"SYNTHETIC replay {index + 1}/120 - green truth / orange detector", fill="white", stroke_width=1, stroke_fill="black")
            if encoder:
                encoder.stdin.write(annotated.tobytes())
            if index in (0, 24, 48, 64, 80, 104, 119):
                annotated.save(output / f"replay_frame_{index:03}.jpg", quality=90)
    finally:
        if encoder:
            encoder.stdin.close()
            error = encoder.stderr.read().decode("utf-8", errors="replace")
            if encoder.wait() != 0:
                encoder_error = error
    switches, fragments = {}, {}
    for object_id, items in observations.items():
        switches[object_id] = sum(a["track_id"] != b["track_id"] and b["frame"] - a["frame"] <= 10
                                  for a, b in zip(items, items[1:]))
        fragments[object_id] = sorted({item["track_id"] for item in items})
    result = {"frame_count": 120, "simulated_fps": 8, "seed": SEED,
              "source_rows": [row for row, _ in sources], "negative_background": background_path,
              "negative_background_status": "included" if background else "skipped: public eligible data not available",
              "totals": dict(totals), "identity_switches_per_object": switches, "track_ids_per_object": fragments,
              "classification_count": len(classification_records), "classifications": classification_records,
              "empty_tracker_at_end": tracker.snapshot()["counts"]["active"] == 0,
              "video": video_path.name if encoder and not encoder_error else None, "encoder_error": encoder_error,
              "frames": records,
              "limitations": ["Synthetic backgrounds/leaves; 8fps timestamps are simulated, not a measured PC or Pi throughput.",
                              "Uses detector, capture_ready and LeafTracker primitives with one classification per track.",
                              "Does not reproduce the live queue, appearance invalidation or a future learned verifier.",
                              "IoU>=0.3 is fixed a priori; ground truth uses visible alpha bbox, including occlusion effects."]}
    write_json(output / "replay_metrics.json", result)
    return {key: value for key, value in result.items() if key not in ("frames", "classifications", "source_rows")}


def extend_negative_suite(source, negative_manifest, threads):
    """Re-evaluate newly audited negative data without repeating classifier tests."""
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    rows = json.loads((source / "validation_rows_snapshot.json").read_text(encoding="utf-8"))
    classifiers = {}
    for name, metadata in protocol["model_snapshots"].items():
        model = source / name / "model.onnx"
        config = model.with_name("config.json")
        if digest(model) != metadata["sha256"] or digest(config) != metadata["config_sha256"]:
            raise RuntimeError(f"Modified frozen model snapshot: {name}")
        classifiers[name] = Classifier(model, config, threads)
    detector_path = source / "detector.onnx"
    if digest(detector_path) != protocol["detector_sha256"]:
        raise RuntimeError("Modified frozen detector snapshot")
    classes = classifiers["baseline"].config["classes"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "reports" / f"robustness_{stamp}_negative_extension"
    output.mkdir(parents=True, exist_ok=False)
    negatives, metadata = read_negatives(negative_manifest)
    write_json(output / "negative_rows_snapshot.json", negatives)
    detector = FastLeafDetector(detector_path, threads=threads)
    metrics = evaluate_negatives(negatives, detector, classifiers, classes, output)
    needed = [next(row for row in rows if row["label"] == label) for label in classes]
    assets = {row["path"]: leaf_asset(row)[1] for row in needed}
    replay_summary = replay(rows, assets, detector, classifiers, classes, output, negatives, shutil.which("ffmpeg"))
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
              "source_frozen_report": str(source.resolve()), "source_protocol_sha256": digest(source / "protocol.json"),
              "classifier_conditions_reused_from": str((source / "classifier_metrics.json").resolve()),
              "negative_manifest": metadata, "negative_proposals": metrics, "replay": replay_summary,
              "limits": "Negative-data extension only; no classifier re-evaluation, training, selection, user photos or deployment."}
    write_json(output / "summary.json", report)
    print(json.dumps({"output": str(output), **report}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "models/potato-mobilenet-v2-onnx/model.onnx")
    parser.add_argument("--candidate", type=Path, default=ROOT / "models/potato-robust-candidate/model.onnx")
    parser.add_argument("--detector", type=Path, default=ROOT / "models/leaf-yolo/leaf_yolo_320.onnx")
    parser.add_argument("--negative-manifest", type=Path, default=ROOT / "data/metadata/nonleaf_manifest.json")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--extend-negative-suite", type=Path,
                        help="Existing frozen report directory; run only new negative data and replay")
    args = parser.parse_args()
    if args.extend_negative_suite:
        extend_negative_suite(args.extend_negative_suite, args.negative_manifest, max(1, args.threads))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "reports" / f"robustness_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    model_paths, snapshots = {}, {}
    for name, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        folder = output / name
        folder.mkdir()
        shutil.copyfile(path, folder / "model.onnx")
        shutil.copyfile(path.with_name("config.json"), folder / "config.json")
        model_paths[name] = folder / "model.onnx"
        snapshots[name] = {"original_path": str(path), "sha256": digest(folder / "model.onnx"),
                           "config_sha256": digest(folder / "config.json")}
    detector_snapshot = output / "detector.onnx"
    shutil.copyfile(args.detector, detector_snapshot)
    classifiers = {name: Classifier(path, path.with_name("config.json"), max(1, args.threads))
                   for name, path in model_paths.items()}
    classes = classifiers["baseline"].config["classes"]
    if classifiers["candidate"].config["classes"] != classes:
        raise RuntimeError("Model class orders differ")
    manifest_path = ROOT / "data/metadata/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    partitions = {split: [row for row in manifest if row["split"] == split] for split in ("train", "val", "test")}
    for key in ("group_id", "sha256", "pixel_sha256"):
        groups = {split: {row[key] for row in rows} for split, rows in partitions.items()}
        if any(groups[a] & groups[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise RuntimeError(f"Split leakage: {key}")
    rows = sorted(partitions["val"], key=lambda row: row["path"])
    write_json(output / "validation_rows_snapshot.json", rows)
    write_json(output / "protocol.json", {
        "seed": SEED, "conditions": list(CONDITIONS), "model_snapshots": snapshots,
        "detector_sha256": digest(detector_snapshot),
        "selection": "All existing validation images; no test images or user photos; no seed search or retraining.",
        "alpha": "train_robust_classifier.foreground(segmented, original_RGB); only boundary-connected black is transparent.",
        "condition_details": {"backgrounds": "Fixed path-hash palette and low-frequency texture; max leaf side206 in256 canvas",
                              "perspective_scale": "Fixed perspective corner fractions, ±18degree rotation, max side166, off-center12,-9",
                              "motion_blur": "Normalized 9-tap linear exposure kernel; direction fixed by path hash",
                              "gamma_rgb": "Fixed warm/cool channel gains and gamma1.20/.78, same rule for every class",
                              "jpeg_occlusion": "JPEGquality42+one rectangle; leaf-relative size17%x28%, source-alpha occlusion recorded"},
        "manifest_sha256": digest(manifest_path),
        "helper_sha256": {name: digest(ROOT / name) for name in (
            "train_robust_classifier.py", "fast_leaf_detector.py", "rover_motion.py", "leaf_tracks.py")},
        "test_image_files_read": False, "user_image_files_read": False,
    })
    probabilities = {name: {condition: [] for condition in CONDITIONS} for name in classifiers}
    assets, source_snapshots, previews = {}, {}, {}
    preview_sources = {label: next(row["path"] for row in rows if row["label"] == label) for label in classes}
    prediction_path = output / "classifier_predictions.jsonl"
    started = time.perf_counter()
    with prediction_path.open("w", encoding="utf-8") as predictions:
        for index, row in enumerate(rows):
            rgb, asset, metadata = leaf_asset(row)
            assets[row["path"]] = asset
            source_snapshots[row["path"]] = metadata
            for condition in CONDITIONS:
                image, detail = compose_condition(rgb, asset, row, condition)
                if preview_sources[row["label"]] == row["path"]:
                    previews[(row["label"], condition)] = image.copy()
                results = {}
                for name, model in classifiers.items():
                    scores = model.predict(image, clean=condition == "clean")
                    probabilities[name][condition].append(scores)
                    results[name] = {"scores": scores.tolist(), "top_class": classes[int(scores.argmax())]}
                predictions.write(json.dumps({"path": row["path"], "group_id": row["group_id"], "label": row["label"],
                                               "condition": condition, "seed": fixed_seed(row["path"]),
                                               "transformation": detail, "results": results}) + "\n")
            if (index + 1) % 40 == 0:
                print(json.dumps({"stage": "classifier", "images_done": index + 1, "images_total": len(rows)}), flush=True)
    write_json(output / "source_snapshots.json", source_snapshots)
    summaries = {name: {condition: grouped_metrics(rows, scores, classes) for condition, scores in conditions.items()}
                 for name, conditions in probabilities.items()}
    comparison = {}
    for condition in CONDITIONS:
        baseline = summaries["baseline"][condition]["images"]
        candidate = summaries["candidate"][condition]["images"]
        comparison[condition] = {"baseline_accuracy": baseline["accuracy"], "candidate_accuracy": candidate["accuracy"],
                                 "accuracy_delta": candidate["accuracy"] - baseline["accuracy"],
                                 "baseline_macro_f1": baseline["macro_f1"], "candidate_macro_f1": candidate["macro_f1"],
                                 "macro_f1_delta": candidate["macro_f1"] - baseline["macro_f1"],
                                 "baseline_confident_errors": baseline["high_confidence_errors"],
                                 "candidate_confident_errors": candidate["high_confidence_errors"],
                                 "class_recall_delta": {label: candidate["per_class"][label]["recall"] - baseline["per_class"][label]["recall"]
                                                        for label in classes}}
    write_json(output / "classifier_metrics.json", summaries)
    preview = Image.new("RGB", (6 * 160, 3 * 190), "#17222c")
    draw = ImageDraw.Draw(preview)
    for ci, condition in enumerate(CONDITIONS):
        for li, label in enumerate(classes):
            thumb = ImageOps.pad(previews[(label, condition)], (154, 154), color=(128, 128, 128))
            preview.paste(thumb, (ci * 160 + 3, li * 190 + 26))
            draw.text((ci * 160 + 3, li * 190 + 3), condition[:20], fill="white")
            draw.text((ci * 160 + 3, li * 190 + 179), label.split("___")[-1], fill="white")
    preview.save(output / "condition_preview.jpg", quality=92)
    negative_rows, negative_metadata = read_negatives(args.negative_manifest)
    write_json(output / "negative_rows_snapshot.json", negative_rows)
    detector = FastLeafDetector(detector_snapshot, threads=max(1, args.threads))
    negatives = evaluate_negatives(negative_rows, detector, classifiers, classes, output) if negative_rows else None
    replay_summary = None
    if not args.skip_replay:
        print(json.dumps({"stage": "replay", "frames": 120}), flush=True)
        replay_summary = replay(rows, assets, detector, classifiers, classes, output, negative_rows, shutil.which("ffmpeg"))
    changes = [name for name, original_path in (("baseline", args.baseline), ("candidate", args.candidate))
               if digest(original_path) != snapshots[name]["sha256"]]
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "output": str(output), "seed": SEED,
              "validation_images": len(rows), "validation_groups": len({row["group_id"] for row in rows}),
              "classes": classes, "model_snapshots": snapshots, "source_model_files_changed_during_run": changes,
              "classifier_conditions": comparison, "negative_manifest": negative_metadata, "negative_proposals": negatives,
              "replay": replay_summary, "elapsed_seconds_pc": time.perf_counter() - started,
              "findings": [f"{condition}: candidate macroF1 delta {values['macro_f1_delta']:+.4f}; confident errors {values['baseline_confident_errors']}->{values['candidate_confident_errors']}"
                           for condition, values in comparison.items()],
              "limitations": ["Existing validation split has already been used for candidate selection; this is diagnostic reuse.",
                              "Only 6 independent healthy leaf groups in validation; image-level results are correlated.",
                              "Compositions and transformations do not establish accuracy on printed leaves, camera video or field plants.",
                              "Three-way disease classifiers lack a learned unknown/non-leaf class; confidence is uncalibrated.",
                              "No test images, user camera images, training, deployment or model selection were performed."]}
    write_json(output / "summary.json", report)
    print(json.dumps({"output": str(output), "conditions": comparison, "negative_proposals": negatives,
                      "replay": replay_summary, "elapsed_seconds_pc": report["elapsed_seconds_pc"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
