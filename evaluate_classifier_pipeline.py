"""Fixed classifier-only comparison on already-opened public validation.

Pre-inference policy: 300 synthetic val scenes (60 empty,480 leaf boxes) plus
39 reviewed person-val crops. The detector, .95 verifier, quality checks and
one-to-one GT association run ONCE and are shared by both classifiers. No new
transformations, print matcher, reserved pixels, training, Pi or deployment.

PASS requires identical upstream decisions, no increase in total confident
wrong diagnoses or wrong diagnoses on true-healthy GT, and no morphology's
correct-eligible diagnosis rate falling more than .03. The denominator is
GT quality eligibility, never a classifier's acceptance. Abstention is not a
correct diagnosis. Thresholds remain softmax>=.7 and margin>=.2. All frames,
including unmatched proposals and negatives, count toward confident errors.
These reused synthetic diagnostics are not independent field accuracy.

--candidate, --report (finished eligible training report), --output required.
The eligibility and both export parity gates are checked BEFORE image loading
or inference. Existing evaluator helpers define preprocessing/IoU/aggregation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from PIL import Image, ImageOps

import evaluate_potato_detector as existing

ROOT = Path(__file__).resolve().parent
SYNTHETIC_SHA = "0fd32e637ecf8e67083f30dd8973b60e4d02d786d9543f1cc3443c39740b0506"
EXPECTED_CANDIDATE = ROOT / "models/potato-print-head-candidate-20261026/model.onnx"
POLICY = {"synthetic_val_frames": 300, "empty_synthetic_frames": 60, "synthetic_gt": 480,
          "natural_person_val_frames": 39, "iou": .5, "min_softmax": .7, "min_margin": .2,
          "same_detector_gate_quality": True, "total_confident_wrong_max_increase": 0,
          "true_healthy_confident_wrong_max_increase": 0, "per_class_correct_eligible_max_loss": .03,
          "healthy_wrong_definition": "Matched true-healthy GT confidently diagnosed as either disease",
          "test_pixels_used": False, "scope": "Reused grouped public validation; no field claim"}


def require_eligible(candidate, report_path):
    """Pure file/metadata checks; no ORT session or image is created here."""
    if candidate.resolve() != EXPECTED_CANDIDATE.resolve():
        raise ValueError("Only the isolated frozen head candidate is permitted")
    report = existing.read_json(report_path)
    if report.get("status") != "ELIGIBLE_OFFLINE_ONLY" or report.get("public_guards_passed") is not True:
        raise ValueError("Training candidate is not eligible; pipeline evaluation remains NOT_EXECUTED")
    config = existing.read_json(candidate.with_name("config.json"))
    export = report.get("export") or {}
    parity = export.get("all_public_decision_parity") or {}
    conditions = {"clean", "backgrounds", "perspective_scale", "motion_blur", "gamma_rgb", "jpeg_occlusion"}
    parity_rows = parity.get("conditions", {})
    logit_error = export.get("max_absolute_logit_difference", float("inf"))
    probability_error = parity.get("max_probability_difference", float("inf"))
    if (report.get("status") != "ELIGIBLE_OFFLINE_ONLY" or report.get("public_guards_passed") is not True
            or report.get("config", {}).get("status") != "ELIGIBLE_OFFLINE_ONLY"
            or config.get("status") != "ELIGIBLE_OFFLINE_ONLY" or config.get("public_guards_passed") is not True
            or export.get("passed") is not True or export.get("all_argmax_match") is not True
            or not math.isfinite(logit_error) or logit_error > 1e-4 or len(export.get("comparisons", [])) != 24
            or parity.get("passed") is not True or parity.get("inputs") != 1968
            or set(parity_rows) != conditions or not math.isfinite(probability_error) or probability_error > 1e-4
            or any(row.get("inputs") != 328 or row.get("argmax_mismatches") != 0 or row.get("decision_mismatches") != 0
                   for row in parity_rows.values())
            or parity.get("policy") != {"softmax": .7, "margin": .2}
            or report.get("test_pixels_used") is not False or report.get("user_fit_used_for_selection") is not False):
        raise ValueError("Candidate training guards and both export parities must PASS before evaluation")
    checksum = existing.digest(candidate)
    if any(value != checksum for value in (config.get("onnx_sha256"), export.get("onnx_sha256"),
                                           report.get("config", {}).get("onnx_sha256"))):
        raise ValueError("Candidate ONNX does not match the eligible frozen report")
    if Path(report.get("output", "")).resolve() != candidate.parent.resolve():
        raise ValueError("Report does not authorize this candidate directory")
    for path, expected in report["config"]["hashes"].items():
        if existing.digest(path) != expected:
            raise ValueError("Frozen training source or input changed")
    return report, config


def load_fixed_validation(dataset, persons):
    if dataset.resolve() != (ROOT / "data/potato_detection_v1").resolve() or existing.digest(dataset / "manifest.json") != SYNTHETIC_SHA:
        raise ValueError("Only the frozen existing synthetic validation dataset is allowed")
    synthetic, natural = existing.load_validation(dataset, persons)
    if sum(len(row["instances"]) for row in synthetic) != 480:
        raise ValueError("Expected480 synthetic validation GT boxes")
    for row in synthetic:
        if row["split"] != "val" or row["background"]["split"] != "val" or any(
                item["source"]["split"] != "val" for item in row["instances"]):
            raise ValueError("Synthetic frame includes a non-validation source")
    for row in natural:
        if row["split"] != "val" or any(term in row["path"].lower() for term in ("future_test", "/test/", "\\test\\")):
            raise ValueError("Only already-opened natural validation rows are allowed")
    return synthetic, natural


class PairedDisease:
    def __init__(self, baseline, candidate):
        self.baseline, self.candidate = baseline, candidate

    def predict(self, crop):
        return {**self.baseline.predict(crop), "_candidate": self.candidate.predict(crop)}


def upstream_signature(detections):
    return [{key: value for key, value in row.items() if key not in ("prediction", "stage")}
            for row in detections]


def analyze_pair(image, detector, verifier, pair):
    detections, timings = existing.analyze(image, detector, verifier, pair)
    branches = {"baseline": [], "candidate": []}
    for detection in detections:
        for branch in branches:
            row = dict(detection)
            if "prediction" in row:
                predictions = row["prediction"]
                row["prediction"] = ({key: value for key, value in predictions.items() if key != "_candidate"}
                                     if branch == "baseline" else predictions["_candidate"])
                row["stage"] = "diagnosed" if row["prediction"]["certain"] else "uncertain"
            branches[branch].append(row)
    identical = upstream_signature(branches["baseline"]) == upstream_signature(branches["candidate"])
    if not identical:
        raise RuntimeError("Classifier evaluation changed shared detection/gate/quality decisions")
    return branches, timings


def extra_counts(frames):
    healthy_wrong = false_healthy = 0
    for frame in frames:
        for match in frame["matches"]:
            truth = frame["gt"][match["gt"]]["class"]
            detection = frame["detections"][match["prediction"]]
            if detection["stage"] != "diagnosed":
                continue
            predicted = detection["prediction"]["top_class"]
            healthy_wrong += truth == "Potato___healthy" and predicted != truth
            false_healthy += truth != "Potato___healthy" and predicted == "Potato___healthy"
    return {"true_healthy_confident_wrong": int(healthy_wrong), "disease_confidently_called_healthy": int(false_healthy)}


def summarize(frames):
    synthetic = [row for row in frames if row["set"] == "synthetic"]
    people = [row for row in frames if row["set"] == "natural_person"]
    result = {"synthetic": existing.aggregate(synthetic), "natural_person": existing.aggregate(people),
              "natural_green_person": existing.aggregate([row for row in people if row["green_clothing_confirmed"]]),
              **extra_counts(frames)}
    result["total_confident_wrong"] = sum(result[key]["counts"].get("confident_wrong", 0) for key in ("synthetic", "natural_person"))
    return result


def guards(baseline, candidate, upstream_identical):
    checks = {"detections_gate_quality_identical": {"passed": bool(upstream_identical)}}
    for key in ("total_confident_wrong", "true_healthy_confident_wrong"):
        checks[key] = {"baseline": baseline[key], "candidate": candidate[key], "maximum_increase": 0,
                       "passed": candidate[key] <= baseline[key]}
    for label in existing.CLASSES:
        before = baseline["synthetic"]["by_class"][label]["correct_eligible_rate"]
        after = candidate["synthetic"]["by_class"][label]["correct_eligible_rate"]
        checks[label + "_correct_eligible"] = {"baseline": before, "candidate": after, "minimum_delta": -.03,
            "passed": before is not None and after is not None and after-before >= -.03-1e-12}
    return {"validation_passed": all(value["passed"] for value in checks.values()), "checks": checks,
            "pi_performance": "NOT_MEASURED", "test_used": False, "deploy_authorized": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, help="Finished head training report.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/potato_detection_v1")
    parser.add_argument("--persons", type=Path, default=ROOT / "data/green_person_v2/final_manifest.json")
    args = parser.parse_args(argv)
    candidate_report, config = require_eligible(args.candidate, args.report)  # FIRST: no pixels/inference before PASS.
    paths = {"baseline": ROOT / "models/leaf-yolo/leaf_yolo_320.onnx",
             "verifier": ROOT / "models/leaf-verifier-candidate/model.onnx",
             "disease": ROOT / "models/potato-robust-candidate/model.onnx"}
    paths.update(verifier_config=paths["verifier"].with_name("config.json"), disease_config=paths["disease"].with_name("config.json"))
    if any(existing.digest(path) != existing.FROZEN_PIPELINE[key] for key, path in paths.items()):
        raise ValueError("Production reference models/configs changed")
    baseline_config = existing.read_json(paths["disease_config"])
    for key in ("classes", "mean", "std", "resize_short_edge", "center_crop", "square_pad", "square_pad_color", "crop_margin"):
        if config.get(key) != baseline_config.get(key):
            raise ValueError("Candidate changed classifier interface/preprocessing")
    synthetic, natural = load_fixed_validation(args.dataset, args.persons)
    protected = [*paths.values(), args.candidate, args.candidate.with_name("config.json"), args.report, Path(__file__),
                 ROOT / "evaluate_potato_detector.py", ROOT / "fast_leaf_detector.py", ROOT / "leaf_verifier.py",
                 ROOT / "rover_motion.py", ROOT / "benchmark_pi_models.py", args.dataset / "manifest.json",
                 args.dataset / "training_readiness.json", args.persons]
    hashes = {str(path.resolve()): existing.digest(path) for path in protected}
    for row in synthetic:
        hashes.update({str(existing.checked_path(args.dataset, row[field])): row[checksum] for field, checksum in
                      (("image", "image_sha256"), ("label", "label_sha256"), ("masks", "mask_sha256"))})
    hashes.update({str(existing.checked_path(ROOT, row["path"])): row["sha256"] for row in natural})
    if not args.output.resolve().is_relative_to((ROOT / "reports").resolve()):
        raise ValueError("Evaluation output must stay inside project reports")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "policy": POLICY, "hashes": hashes,
                "candidate_training_report": str(args.report.resolve()), "status": "FROZEN_BEFORE_INFERENCE"}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    detector = existing.FastLeafDetector(paths["baseline"], threads=2)
    verifier = existing.LeafVerifier(paths["verifier"])
    pair = PairedDisease(existing.DiseaseModel(paths["disease"]), existing.DiseaseModel(args.candidate))
    records = {"baseline": [], "candidate": []}
    started = time.perf_counter()
    with (args.output / "paired_frames.jsonl").open("x", encoding="utf-8") as stream:
        for index, (category, row) in enumerate([("synthetic", item) for item in synthetic] + [("natural_person", item) for item in natural]):
            path = args.dataset / row["image"] if category == "synthetic" else ROOT / row["path"]
            with Image.open(path) as source:
                image = source.convert("RGB")
            if category == "natural_person":
                image = ImageOps.contain(image, (640, 360), Image.Resampling.BILINEAR)
            gt = [{"box": item["bbox_xyxy"], "class": item["source"]["label"], "group_id": item["source"]["group_id"],
                   "eligible": bool(existing.capture_ready(image, item["bbox_xyxy"], existing.leaf_crop(image, item["bbox_xyxy"])))}
                  for item in row.get("instances", [])]
            branches, timings = analyze_pair(image, detector, verifier, pair)
            matches = existing.match_boxes([d["box"] for d in branches["baseline"]], [g["box"] for g in gt])
            frame = {"set": category, "image": str(path.relative_to(ROOT)), "gt": gt, "matches": matches, "timings": timings,
                     "upstream_identical": True, "green_clothing_confirmed": row.get("green_clothing_confirmed", False)}
            stream.write(json.dumps({**frame, "classifiers": branches}) + "\n")
            for key in records:
                records[key].append({**frame, "detections": branches[key]})
            if index % 50 == 0:
                print(json.dumps({"paired_frames_complete": index+1, "total": 339}), flush=True)
    if any(existing.digest(path) != expected for path, expected in hashes.items()):
        raise RuntimeError("Protected pipeline, model, source or input changed during evaluation")
    baseline, candidate = summarize(records["baseline"]), summarize(records["candidate"])
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "policy": POLICY, "baseline": baseline, "candidate": candidate,
               "guards": guards(baseline, candidate, True), "hashes": hashes, "elapsed_seconds": time.perf_counter()-started,
               "candidate_training_status": candidate_report["status"], "test_used": False, "pi_accessed": False, "deployed": False,
               "limitations": ["Reused synthetic validation views share82 source-leaf groups; not independent field observations",
                   "Printed-source identity is unknown and may overlap public source leaves",
                   "Natural negatives are39 previously opened author groups, only2 confirmed green garments",
                   "Static PC pipeline does not test temporal tracking, motion cadence, Pi resources or biological field accuracy"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "guards": summary["guards"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
