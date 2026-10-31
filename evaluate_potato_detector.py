"""Compare two frozen detectors on validation only, with the production pipeline.

No training, threshold changes, test pixels, user images, Pi access or deployment.
The acceptance limits come from docs/POTATO_DETECTOR_PROTOCOL.md. Frame-level
results are diagnostic; correlated synthetic views are not independent leaves.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps

from benchmark_pi_models import DiseaseModel
from fast_leaf_detector import FastLeafDetector
from leaf_verifier import LeafVerifier
from rover_motion import capture_ready, leaf_crop

ROOT = Path(__file__).resolve().parent
CLASSES = ("Potato___Early_blight", "Potato___Late_blight", "Potato___healthy")
PERSON_MANIFEST_SHA256 = "1ab9da6c25c11ca646bca7111810705cb75f8218ff9687955abd0912040d3687"
FROZEN_PIPELINE = {
    "baseline": "3e58ae60d202e07c81fed21eea6253047ac6f6eb1752499cca99504dda5d7c3c",
    "verifier": "25439c84f695e238d03a27ae1e0013be17b96b1c6911f07ffe225fd7114cde03",
    "verifier_config": "a47afeb970a90872464ea16dd349033a9e4437015ff4b65d8dd127c3564e8276",
    "disease": "a0fff51ac90c7e8f29dd634a5c9e81be1236f241b4cc94eebe3b3687ae19eec8",
    "disease_config": "30ef816f0f70eb3a318d42d13aa97a6dfb2bfb96d2591a657f790ea84e5d67b2",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def checked_path(root, value):
    path = (root / value).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Manifest path escapes its root")
    return path


def iou(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0., x2-x1) * max(0., y2-y1)
    a = max(0., first[2]-first[0]) * max(0., first[3]-first[1])
    b = max(0., second[2]-second[0]) * max(0., second[3]-second[1])
    return overlap / (a+b-overlap) if a+b-overlap > 0 else 0.


def match_boxes(predictions, ground_truth, threshold=.5):
    """Maximum-cardinality one-to-one match; break ties by total IoU.

    There are at most three GT per synthetic frame. A bitmask DP avoids the
    incorrect greedy match which can sacrifice a second valid leaf.
    """
    if len(ground_truth) > 12:
        raise ValueError("GT count exceeds the bounded matching protocol")
    states = {0: (0., ())}
    for p, prediction in enumerate(predictions):
        next_states = dict(states)
        for mask, (score, matches) in states.items():
            for g, target in enumerate(ground_truth):
                overlap = iou(prediction, target)
                if mask & (1 << g) or overlap < threshold:
                    continue
                new_mask = mask | (1 << g)
                candidate = (score+overlap, matches+((p, g, overlap),))
                old = next_states.get(new_mask)
                if old is None or candidate[0] > old[0]:
                    next_states[new_mask] = candidate
        states = next_states
    _, (_, matches) = max(states.items(), key=lambda item: (item[0].bit_count(), item[1][0]))
    return [{"prediction": p, "gt": g, "iou": float(overlap)} for p, g, overlap in matches]


def load_validation(dataset, persons):
    manifest = dataset / "manifest.json"
    readiness = read_json(dataset / "training_readiness.json")
    if readiness.get("can_train") is not True or readiness.get("manifest_sha256") != digest(manifest):
        raise ValueError("The exact synthetic manifest has not passed QA")
    synthetic = [row for row in read_json(manifest) if row["split"] == "val"]
    if len(synthetic) != 300 or sum(row["negative"] for row in synthetic) != 60:
        raise ValueError("Expected exactly300 validation frames including60 negatives")
    if digest(persons) != PERSON_MANIFEST_SHA256:
        raise ValueError("Natural validation manifest differs from the preregistered frozen version")
    natural = [row for row in read_json(persons) if row["split"] == "val"]
    if len(natural) != 39 or len({row["group_id"] for row in natural}) != 39:
        raise ValueError("Expected39 frozen natural validation author groups")
    for row in natural:
        if row.get("is_leaf") is not False or not row.get("audit_status", "").startswith("visually_reviewed"):
            raise ValueError("Natural validation contains an unreviewed or leaf crop")
    for row in synthetic:
        for field, checksum in (("image", "image_sha256"), ("label", "label_sha256"), ("masks", "mask_sha256")):
            if digest(checked_path(dataset, row[field])) != row[checksum]:
                raise ValueError("Frozen synthetic input changed")
    for row in natural:
        if digest(checked_path(ROOT, row["path"])) != row["sha256"]:
            raise ValueError("Frozen natural crop changed")
    return synthetic, natural


def analyze(image, detector, verifier, disease):
    started = time.perf_counter()
    detections = detector.detect(image)
    detector_ms = (time.perf_counter()-started)*1000
    results = []
    for detection in detections:
        result = dict(detection)
        crop = leaf_crop(image, detection["box"])
        result["capture_ready"] = bool(capture_ready(image, detection["box"], crop))
        result["stage"] = "capture_not_ready"
        if result["capture_ready"]:
            verification = verifier.verify(crop)
            result["verification"] = verification
            result["stage"] = "verifier_rejected"
            if verification["accepted"]:
                prediction = disease.predict(crop)
                result["prediction"] = prediction
                result["stage"] = "diagnosed" if prediction["certain"] else "uncertain"
        results.append(result)
    return results, {"detector_ms": detector_ms, "pipeline_ms": (time.perf_counter()-started)*1000}


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def aggregate(frames):
    counts = Counter()
    by_class = {label: Counter() for label in CLASSES}
    overlaps, groups = [], set()
    for frame in frames:
        detections, gt, matches = frame["detections"], frame["gt"], frame["matches"]
        counts["frames"] += 1
        counts["proposals"] += len(detections)
        counts["gt"] += len(gt)
        counts["matched"] += len(matches)
        counts["unmatched_proposals"] += len(detections)-len(matches)
        counts["negative_frames"] += int(not gt)
        counts["negative_proposals"] += len(detections) if not gt else 0
        counts["confident_wrong"] += sum(d["stage"] == "diagnosed" for d in detections)
        by_gt = {m["gt"]: m for m in matches}
        for g, target in enumerate(gt):
            label = target["class"]
            groups.add(target["group_id"])
            by_class[label]["gt"] += 1
            eligible = bool(target["eligible"])
            counts["eligible_gt"] += eligible
            by_class[label]["eligible_gt"] += eligible
            match = by_gt.get(g)
            correct = False
            if match:
                overlaps.append(match["iou"])
                counts["eligible_matched"] += eligible
                by_class[label]["matched"] += 1
                by_class[label]["eligible_matched"] += eligible
                detected = detections[match["prediction"]]
                counts["matched_diagnosed"] += detected["stage"] == "diagnosed"
                correct = (detected["stage"] == "diagnosed" and detected["prediction"]["top_class"] == label)
            if correct:
                counts["confident_wrong"] -= 1
                counts["correct_diagnoses"] += 1
                counts["eligible_correct"] += eligible
                by_class[label]["correct_diagnoses"] += 1
                by_class[label]["eligible_correct"] += eligible
        for detection in detections:
            counts[f"stage_{detection['stage']}"] += 1
    result = {"counts": dict(counts), "leaf_source_groups": len(groups),
              "recall_all": rate(counts["matched"], counts["gt"]),
              "recall_eligible": rate(counts["eligible_matched"], counts["eligible_gt"]),
              "precision": rate(counts["matched"], counts["proposals"]),
              "correct_eligible_rate": rate(counts["eligible_correct"], counts["eligible_gt"]),
              "diagnostic_coverage": rate(counts["matched_diagnosed"], counts["gt"]),
              "diagnosed_proposals_per_gt": rate(counts["stage_diagnosed"], counts["gt"]),
              "matched_iou": {"n": len(overlaps), "p50": float(np.percentile(overlaps, 50)) if overlaps else None,
                              "p05": float(np.percentile(overlaps, 5)) if overlaps else None},
              "by_class": {}}
    for label, c in by_class.items():
        result["by_class"][label] = {"counts": dict(c), "recall_eligible": rate(c["eligible_matched"], c["eligible_gt"]),
                                    "correct_eligible_rate": rate(c["eligible_correct"], c["eligible_gt"])}
    return result


def evaluate_guards(baseline, candidate):
    checks = {}
    def delta_guard(key, before, after, tolerance):
        checks[key] = {"baseline": before, "candidate": after, "minimum_delta": -tolerance,
                       "passed": before is not None and after is not None and after-before >= -tolerance-1e-12}
    b, c = baseline["synthetic"], candidate["synthetic"]
    delta_guard("all_gt_recall", b["recall_all"], c["recall_all"], .02)
    delta_guard("eligible_recall", b["recall_eligible"], c["recall_eligible"], .01)
    delta_guard("eligible_correct_diagnoses", b["correct_eligible_rate"], c["correct_eligible_rate"], .01)
    for label in CLASSES:
        for metric in ("recall_eligible", "correct_eligible_rate"):
            delta_guard(f"{label}_{metric}", b["by_class"][label][metric], c["by_class"][label][metric], .03)
    for dataset, metric in (("synthetic", "negative_proposals"), ("synthetic", "confident_wrong"),
                            ("natural_person", "proposals"), ("natural_person", "confident_wrong")):
        before, after = baseline[dataset]["counts"].get(metric, 0), candidate[dataset]["counts"].get(metric, 0)
        checks[f"{dataset}_{metric}"] = {"baseline": before, "candidate": after, "passed": after <= before}
    return {"validation_passed": all(row["passed"] for row in checks.values()), "checks": checks,
            "pi_performance": "NOT_YET_MEASURED", "test_used": False, "deploy_authorized": False}


def evaluate_model(path, synthetic, natural, dataset, verifier, disease, destination):
    detector = FastLeafDetector(path, threads=2)
    records, people = [], []
    with destination.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(synthetic):
            with Image.open(dataset / row["image"]) as source:
                image = source.convert("RGB")
            gt = [{"box": item["bbox_xyxy"], "class": item["source"]["label"],
                   "group_id": item["source"]["group_id"],
                   "eligible": bool(capture_ready(image, item["bbox_xyxy"], leaf_crop(image, item["bbox_xyxy"]))) }
                  for item in row["instances"]]
            detections, timings = analyze(image, detector, verifier, disease)
            result = {"set": "synthetic", "image": row["image"], "gt": gt, "detections": detections,
                      "matches": match_boxes([d["box"] for d in detections], [g["box"] for g in gt]), "timings": timings}
            records.append(result)
            stream.write(json.dumps(result)+"\n")
            if index % 50 == 0:
                print(json.dumps({"model": str(path), "validation_frames": index+1}), flush=True)
        for row in natural:
            with Image.open(ROOT / row["path"]) as source:
                # Preserve crop aspect ratio and fit the production frame size.
                image = ImageOps.contain(source.convert("RGB"), (640, 360), Image.Resampling.BILINEAR)
            detections, timings = analyze(image, detector, verifier, disease)
            result = {"set": "natural_person", "image": row["path"], "group_id": row["group_id"],
                      "green_clothing_confirmed": row["green_clothing_confirmed"], "gt": [],
                      "detections": detections, "matches": [], "timings": timings}
            people.append(result)
            stream.write(json.dumps(result)+"\n")
    return {"synthetic": aggregate(records), "natural_person": aggregate(people),
            "natural_green_person": aggregate([row for row in people if row["green_clothing_confirmed"]])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT/"models/leaf-yolo/leaf_yolo_320.onnx")
    parser.add_argument("--dataset", type=Path, default=ROOT/"data/potato_detection_v1")
    parser.add_argument("--persons", type=Path, default=ROOT/"data/green_person_v2/final_manifest.json")
    parser.add_argument("--verifier", type=Path, default=ROOT/"models/leaf-verifier-candidate/model.onnx")
    parser.add_argument("--disease", type=Path, default=ROOT/"models/potato-robust-candidate/model.onnx")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    pipeline_paths = {"baseline": args.baseline, "verifier": args.verifier, "disease": args.disease,
                      "verifier_config": args.verifier.with_name("config.json"),
                      "disease_config": args.disease.with_name("config.json")}
    if any(digest(path) != FROZEN_PIPELINE[key] for key, path in pipeline_paths.items()):
        raise ValueError("Reference models/configurations differ from the frozen Pi pipeline")
    if read_json(pipeline_paths["verifier_config"])["threshold"] != .95:
        raise ValueError("The frozen verifier threshold must remain .95")
    protected = [args.candidate, args.baseline, args.verifier, args.disease, args.verifier.with_name("config.json"),
                 args.disease.with_name("config.json"), args.dataset/"manifest.json", args.persons,
                 Path(__file__), ROOT/"fast_leaf_detector.py", ROOT/"rover_motion.py", ROOT/"leaf_verifier.py",
                 ROOT/"benchmark_pi_models.py", ROOT/"docs/POTATO_DETECTOR_PROTOCOL.md"]
    hashes = {str(path.resolve()): digest(path) for path in protected}
    synthetic, natural = load_validation(args.dataset, args.persons)
    candidate_interface = FastLeafDetector(args.candidate)
    if (candidate_interface.input_size != (320, 320)
            or candidate_interface.session.get_outputs()[0].shape != [1, 5, 2100]
            or list(candidate_interface.names.values() if isinstance(candidate_interface.names, dict)
                    else candidate_interface.names) != ["leaf"]):
        raise ValueError("Candidate is not the trained one-class320 raw leaf detector")
    del candidate_interface
    args.output.mkdir(parents=True, exist_ok=False)
    verifier, disease = LeafVerifier(args.verifier), DiseaseModel(args.disease)
    baseline = evaluate_model(args.baseline, synthetic, natural, args.dataset, verifier, disease, args.output/"baseline.jsonl")
    candidate = evaluate_model(args.candidate, synthetic, natural, args.dataset, verifier, disease, args.output/"candidate.jsonl")
    if any(digest(Path(path)) != checksum for path, checksum in hashes.items()):
        raise RuntimeError("Protected source, model or manifest changed during evaluation")
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "baseline": baseline, "candidate": candidate,
               "guards": evaluate_guards(baseline, candidate), "hashes": hashes, "test_used": False, "user_images_used": False,
               "limitations": ["Synthetic reused validation leaves are correlated, not independent field observations",
                               "Natural negatives are39 reviewed author groups and only2 confirmed green garments",
                               "PC static validation does not measure Pi speed or correct moving-track identity",
                               "Per-frame pipeline evaluates every eligible proposal without temporal tracking"]}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "guards": summary["guards"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
