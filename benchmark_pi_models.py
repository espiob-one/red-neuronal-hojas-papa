"""Read-only CPU inference benchmark; no SSH, services, camera control or deployment.

--prepare builds small public validation fixtures on the PC. Normal execution
needs only numpy/Pillow/onnxruntime and the detector/verifier wrappers on the Pi.
Model loading, JPEG decoding, hashing and telemetry reads stay outside inference
timers. This benchmark never selects or modifies models or policy thresholds.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from PIL import Image, ImageOps

from fast_leaf_detector import FastLeafDetector
from leaf_verifier import LeafVerifier


ROOT = Path(__file__).resolve().parent


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def telemetry():
    rss, peak, temperature = None, None, None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1024
            elif line.startswith("VmHWM:"):
                peak = int(line.split()[1]) / 1024
    except OSError:
        pass
    try:
        temperature = int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except (OSError, ValueError):
        pass
    return {"rss_mib": rss, "peak_rss_mib": peak, "temperature_c": temperature}


def throttling():
    try:
        result = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def crop_leaf(image, box, padding=.12):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    bounds = (max(0, int(x1 - dx)), max(0, int(y1 - dy)),
              min(image.width, math.ceil(x2 + dx)), min(image.height, math.ceil(y2 + dy)))
    crop = image.crop(bounds).convert("RGB")
    return ImageOps.pad(crop, (max(crop.size),) * 2, color=(128, 128, 128))


def capture_quality(image, box, crop):
    gray = np.asarray(crop.convert("L").resize((160, 160)), dtype=np.float32)
    lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    sharp = float(lap.var())
    x1, y1, x2, y2 = box
    return (min(x2 - x1, y2 - y1) >= 40 and sharp >= 30 and x1 >= 3 and y1 >= 3
            and x2 <= image.width - 3 and y2 <= image.height - 3), sharp


class DiseaseModel:
    def __init__(self, model):
        import onnxruntime as ort
        self.config_path = model.with_name("config.json")
        self.config = json.loads(self.config_path.read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 1, 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image):
        side, edge = self.config["resize_short_edge"], self.config["center_crop"]
        image = image.resize((side, side), Image.Resampling.BILINEAR)
        offset = (side - edge) // 2
        array = np.asarray(image.crop((offset, offset, offset + edge, offset + edge)), dtype=np.float32) / 255
        array = (array - np.asarray(self.config["mean"], dtype=np.float32)) / np.asarray(self.config["std"], dtype=np.float32)
        tensor = np.ascontiguousarray(array.transpose(2, 0, 1)[None])
        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        exp = np.exp(logits - logits.max())
        scores = exp / exp.sum()
        order = np.argsort(scores)
        return {"top_class": self.config["classes"][int(order[-1])], "score": float(scores[order[-1]]),
                "margin": float(scores[order[-1]] - scores[order[-2]]), "scores": scores.tolist(),
                "certain": bool(scores[order[-1]] >= .7 and scores[order[-1]] - scores[order[-2]] >= .2)}


def timed(function, *args):
    start = time.perf_counter()
    result = function(*args)
    return result, (time.perf_counter() - start) * 1000


def distribution(values):
    values = [float(value) for value in values if value is not None]
    return {"n": len(values), "p50": float(np.percentile(values, 50)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
            "minimum": min(values) if values else None, "maximum": max(values) if values else None}


def prepare_samples(output):
    """PC-only fixture generation; never selects for detectability or correctness."""
    from train_robust_classifier import foreground
    from evaluate_robustness_suite import texture, motion_blur
    rows = json.loads((ROOT / "data/metadata/manifest.json").read_text())
    classes = json.loads((ROOT / "data/metadata/classes.json").read_text())
    negatives = json.loads((ROOT / "data/metadata/nonleaf_manifest.json").read_text(encoding="utf-8-sig"))
    negatives = sorted((row for row in negatives if row["split"] == "val" and row["is_leaf"] is False
                        and row["semantic_role"] == "person_region" and row.get("audit_status", "").startswith("visually_reviewed")),
                       key=lambda row: row["path"])
    unique_scenes = []
    for row in negatives:
        if row["group_id"] not in {item["group_id"] for item in unique_scenes}:
            unique_scenes.append(row)
    if len(unique_scenes) < 2:
        raise RuntimeError("Need two independently reviewed public person validation scenes")
    output.mkdir(parents=True, exist_ok=True)
    samples = []
    for label in classes:
        groups = {}
        for row in sorted((row for row in rows if row["split"] == "val" and row["label"] == label), key=lambda row: row["path"]):
            groups.setdefault(row["group_id"], row)
        selected = list(groups.values())[:6]
        if len(selected) != 6:
            raise RuntimeError(f"Need six validation leaf groups in {label}")
        for view, row in enumerate(selected):
            with Image.open(ROOT / row["path"]) as source:
                original = source.convert("RGB")
            segmented = ROOT / "data/segmented" / label / (Path(row["path"]).stem + "_final_masked.jpg")
            with Image.open(segmented) as source:
                leaf = foreground(source.convert("RGB"), original)
            size = (175, 180, 145, 190, 160, 180)[view]
            ratio = size / max(leaf.size)
            leaf = leaf.resize((round(leaf.width * ratio), round(leaf.height * ratio)), Image.Resampling.BILINEAR)
            position = ((-25, 85), (35, 75), (220, 115), (410, 95), (535, 100), (285, -15))[view]
            background_row = unique_scenes[view % 2] if view in (2, 3) else None
            if background_row:
                with Image.open(ROOT / background_row["path"]) as source:
                    frame = ImageOps.fit(source.convert("RGB"), (640, 360), method=Image.Resampling.BILINEAR)
            else:
                frame = texture((640, 360), f"pi-benchmark-view-{view}")
            frame.paste(leaf, position, leaf)
            blur = view in (3, 4)
            if blur:
                frame = motion_blur(frame, 9, "h")
            bounds = [max(0, position[0]), max(0, position[1]),
                      min(640, position[0] + leaf.width), min(360, position[1] + leaf.height)]
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                raise RuntimeError("Fixture leaf has no visible area")
            crop = crop_leaf(frame, bounds)
            sample_id = f"{len(samples):02d}_{label.split('___')[-1]}_view{view}"
            frame_name, crop_name = sample_id + "_frame.jpg", sample_id + "_crop.jpg"
            frame.save(output / frame_name, quality=90)
            crop.resize((224, 224), Image.Resampling.BILINEAR).save(output / crop_name, quality=90)
            samples.append({"id": sample_id, "frame": frame_name, "crop": crop_name, "expected_class": label,
                            "source_path": row["path"], "source_group": row["group_id"], "source_sha256": row["sha256"],
                            "source_split": "val", "segmented_sha256": file_hash(segmented),
                            "background_path": background_row["path"] if background_row else "deterministic_synthetic_texture",
                            "background_license": background_row.get("license") if background_row else None,
                            "position_xy": list(position), "source_leaf_box_visible": bounds,
                            "linear_motion_blur_kernel": 9 if blur else None,
                            "condition": "motion" if blur else "person_background" if background_row else "off_center_or_border"})
    for row in unique_scenes[:2]:
        with Image.open(ROOT / row["path"]) as source:
            image = source.convert("RGB")
        frame = ImageOps.fit(image, (640, 360), method=Image.Resampling.BILINEAR)
        crop = ImageOps.pad(image, (max(image.size),) * 2, color=(128, 128, 128)).resize((224, 224), Image.Resampling.BILINEAR)
        sample_id = f"{len(samples):02d}_person_nonleaf"
        frame.save(output / (sample_id + "_frame.jpg"), quality=90)
        crop.save(output / (sample_id + "_crop.jpg"), quality=90)
        samples.append({"id": sample_id, "frame": sample_id + "_frame.jpg", "crop": sample_id + "_crop.jpg",
                        "expected_class": None, "source_path": row["path"], "source_group": row["group_id"],
                        "source_sha256": row["sha256"], "source_split": "val", "condition": "person_nonleaf",
                        "license": row.get("license"), "public_provenance": row.get("source_url")})
    for sample in samples:
        sample["frame_sha256"] = file_hash(output / sample["frame"])
        sample["crop_sha256"] = file_hash(output / sample["crop"])
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "sample_count": 20,
                "positive_count_per_class": {label: 6 for label in classes}, "person_negative_count": 2,
                "purpose": "Fixed performance/sanity fixtures only; not model selection or an accuracy holdout.",
                "sources": ["PlantVillage original potato RGB and matched segmented masks; existing validation split",
                            "Visually reviewed COCO panoptic-derived person validation regions; no user pictures"],
                "alpha_policy": "foreground(segmented, original RGB), preserving dark interior tissue",
                "fixtures": samples}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"manifest": str((output / "manifest.json").resolve()), "samples": len(samples),
                      "total_bytes": sum(path.stat().st_size for path in output.iterdir() if path.is_file())}, indent=2))


def benchmark(args):
    for name in ("manifest", "detector", "verifier", "disease", "output"):
        path = getattr(args, name)
        if path is None or not path.is_absolute():
            raise ValueError(f"--{name} must be an absolute path")
    manifest = json.loads(args.manifest.read_text())
    samples = manifest["fixtures"]
    if len(samples) != 20:
        raise ValueError("This protocol requires the frozen 20-sample fixture manifest")
    loaded = []
    for sample in samples:
        images = []
        for key in ("frame", "crop"):
            path = args.manifest.parent / sample[key]
            if file_hash(path) != sample[key + "_sha256"]:
                raise ValueError(f"Sample pixels changed: {sample['id']} {key}")
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        loaded.append((sample, *images))
    hashes = {name: file_hash(getattr(args, name)) for name in ("detector", "verifier", "disease", "manifest")}
    hashes.update(verifier_config=file_hash(args.verifier.with_name("config.json")),
                  disease_config=file_hash(args.disease.with_name("config.json")), script=file_hash(__file__))
    before = {"telemetry": telemetry(), "throttled": throttling()}
    detector = FastLeafDetector(args.detector, threads=2)
    if detector.input_size != (320, 320):
        raise ValueError("This protocol requires the frozen 320x320 detector")
    verifier = LeafVerifier(args.verifier, threads=1)
    disease = DiseaseModel(args.disease)
    after_load = telemetry()
    for index in range(args.warmup):
        _, frame, crop = loaded[index % len(loaded)]
        detector.detect(frame)
        verifier.verify(crop)
        disease.predict(crop)
    records = []
    for repeat in range(args.runs):
        for sample, frame, nominal_crop in loaded:
            start = time.perf_counter()
            detections, detector_ms = timed(detector.detect, frame)
            verified = prediction = None
            pipeline_verifier_ms = pipeline_disease_ms = None
            quality, sharpness = False, None
            if detections:
                detection = max(detections, key=lambda item: item["score"])
                crop = crop_leaf(frame, detection["box"])
                quality, sharpness = capture_quality(frame, detection["box"], crop)
                # Verification is measured even for low-quality detections; only
                # acceptable, physically usable crops reach disease inference.
                verified, pipeline_verifier_ms = timed(verifier.verify, crop)
                if verified["accepted"] and quality:
                    prediction, pipeline_disease_ms = timed(disease.predict, crop)
            pipeline_total_ms = (time.perf_counter() - start) * 1000
            if not detections:
                outcome = "no_detection"
            elif not verified["accepted"]:
                outcome = "rejected_nonleaf"
            elif not quality:
                outcome = "uncertain_capture"
            elif not prediction["certain"]:
                outcome = "uncertain_disease"
            else:
                outcome = "accepted_classification"
            # Independent fixed-crop component probes ensure that each model has
            # 40 observations even when the real pipeline legitimately abstains.
            probe_verifier, probe_verifier_ms = timed(verifier.verify, nominal_crop)
            probe_prediction, probe_disease_ms = timed(disease.predict, nominal_crop)
            records.append({"run": repeat + 1, "sample_id": sample["id"], "expected_class": sample["expected_class"],
                            "condition": sample["condition"], "outcome": outcome, "detections": len(detections),
                            "detector_ms": detector_ms, "pipeline_verifier_ms": pipeline_verifier_ms,
                            "pipeline_classifier_ms": pipeline_disease_ms, "pipeline_total_ms": pipeline_total_ms,
                            "capture_quality": quality, "sharpness": sharpness, "verification": verified,
                            "disease": prediction, "probe_verifier_ms": probe_verifier_ms,
                            "probe_classifier_ms": probe_disease_ms,
                            "controlled_component_sum_ms": detector_ms + probe_verifier_ms + probe_disease_ms,
                            "probe_verification": probe_verifier, "probe_disease": probe_prediction, **telemetry()})
    end = {"telemetry": telemetry(), "throttled": throttling()}
    keys = ("detector_ms", "pipeline_verifier_ms", "pipeline_classifier_ms", "pipeline_total_ms",
            "probe_verifier_ms", "probe_classifier_ms", "controlled_component_sum_ms", "rss_mib", "peak_rss_mib", "temperature_c")
    final_hashes = {name: file_hash(getattr(args, name)) for name in ("detector", "verifier", "disease", "manifest")}
    final_hashes.update(verifier_config=file_hash(args.verifier.with_name("config.json")),
                        disease_config=file_hash(args.disease.with_name("config.json")), script=file_hash(__file__))
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
               "machine": platform.machine(), "inference_provider": "CPUExecutionProvider", "hashes": hashes,
               "inputs_changed_during_measurement": [key for key in hashes if hashes[key] != final_hashes[key]],
               "verifier_threshold": verifier.threshold,
               "threads": {"detector_intra": 2, "verifier_intra": 1, "classifier_intra": 1, "inter_op_each": 1},
               "warmup": args.warmup, "runs": args.runs, "samples_per_run": len(loaded), "measurements": len(records),
               "latencies_memory_temperature": {key: distribution([row[key] for row in records]) for key in keys},
               "outcomes": dict(Counter(row["outcome"] for row in records)),
               "outcomes_by_condition": {condition: dict(Counter(row["outcome"] for row in records if row["condition"] == condition))
                                         for condition in sorted({row["condition"] for row in records})},
               "before_loading": before, "after_loading": after_load, "end": end,
               "notes": ["Loading, image decoding, hashes and telemetry reads are outside inference timers.",
                         "Pipeline total is measured wall time with detection, crop/quality, verification and gated disease inference.",
                         "Probe timings use a fixed public crop independent of detector success; component sum is a sum, not live-stream latency.",
                         "Forty sequential samples give descriptive p50/p95 only; no accuracy or runtime selection was performed.",
                         "No tracking, live camera access, queue, SSH, service changes, power measurement or user photos.",
                         "RSS/temperature may be unavailable on non-Linux hosts; they are reported as null."]}
    write_json(args.output, {"summary": summary, "measurements": records})
    print(json.dumps({"report": str(args.output), **summary}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", type=Path, help="PC only: output directory for public fixtures")
    for name in ("manifest", "detector", "verifier", "disease", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    if args.prepare:
        prepare_samples(args.prepare.resolve())
    else:
        if args.warmup < 0 or args.runs < 1:
            parser.error("Nonnegative warmup and positive runs required")
        benchmark(args)


if __name__ == "__main__":
    main()
