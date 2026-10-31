"""Prepare/freeze a local YOLOv8n leaf-detection experiment; train only after QA.

Default operation writes a reviewable protocol. --train additionally requires
an unchanged protocol and dataset training_readiness.json with can_train=true.
No deployment, existing model replacement, or external services are supported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import setup_yolo_tools as tools

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "data/potato_detection_v1/dataset.yaml"
DEFAULT_PROTOCOL = ROOT / "reports/potato_detector_training_protocol.json"
PROJECT = ROOT / "runs/potato_detector_candidates"
SEED = 2026102204


def export_parity_protocol(manifest):
    """Freeze model-independent validation inputs and acceptance rules."""
    rows = [row for row in manifest if row.get("split") == "val"]
    if len(rows) < 20:
        raise ValueError("Export parity requires at least 20 validation images")
    def rank(row):
        return hashlib.sha256(f"{SEED}:{row['image_sha256']}:{row['image']}".encode()).hexdigest()
    selected = sorted(rows, key=lambda row: (rank(row), row["image"]))[:20]
    samples = [{"image": row["image"], "image_sha256": row["image_sha256"]} for row in selected]
    if len({row["image"] for row in samples}) != 20 or len({row["image_sha256"] for row in samples}) != 20:
        raise ValueError("Export validation inputs must be distinct images")
    return {"selection": "20 lowest SHA256(seed:image_sha256:image_path) from validation manifest",
            "seed": SEED, "split": "val", "samples": samples,
            "preprocessing": "fast_leaf_detector.letterbox RGB float32 NCHW 320x320",
            "decoder": "fast_leaf_detector.decode_output",
            "decode": {"threshold": .45, "iou_threshold": .45, "limit": 8, "min_side": 8.},
            "max_raw_box_absolute_error": .05, "max_raw_score_absolute_error": .001,
            "equal_decoded_counts": True, "equal_decoded_classes": True,
            "min_matched_iou": .999,
            "matching": "one-to-one same-class bipartite matching at required minimum IoU"}


def _dataset_file(value, root):
    path = Path(value)
    if not path.is_absolute():
        path = (ROOT/path) if str(path).replace("\\", "/").startswith("data/") else root/path
    path = path.resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("Dataset entry escapes the frozen dataset directory")
    return path


def dataset_metadata(yaml_path):
    import yaml
    yaml_path = yaml_path.resolve(strict=True)
    dataset_root = yaml_path.parent
    parsed = yaml.safe_load(yaml_path.read_text())
    if not isinstance(parsed, dict) or "download" in parsed or parsed.get("test"):
        raise ValueError("Only a local train/val dataset without download/test entries is allowed")
    names = parsed.get("names")
    if names not in ({0: "leaf"}, {"0": "leaf"}, ["leaf"]) or parsed.get("nc", 1) != 1:
        raise ValueError("Expected exactly one class: leaf, id0")
    declared_root = Path(parsed.get("path", str(dataset_root)))
    if not declared_root.is_absolute():
        declared_root = dataset_root/declared_root
    if declared_root.resolve() != dataset_root:
        raise ValueError("Dataset YAML root differs from its frozen directory")
    for split in ("train", "val"):
        split_path = _dataset_file(parsed.get(split, ""), dataset_root)
        if split_path != (dataset_root/"images"/split).resolve():
            raise ValueError("Expected local images/train and images/val directories")
    manifest_path = dataset_root/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, list):
        raise ValueError("Dataset manifest must be a list of frame records")
    counts = {split: sum(row.get("split") == split for row in manifest) for split in ("train", "val")}
    if counts != {"train": 1800, "val": 300} or sum(counts.values()) != len(manifest):
        raise ValueError("Expected the reviewed 1800 train /300 validation frame dataset")
    return {"yaml": str(yaml_path), "yaml_sha256": tools.digest(yaml_path),
            "manifest": str(manifest_path), "manifest_sha256": tools.digest(manifest_path),
            "counts": counts, "names": {"0": "leaf"}}, manifest


def train_parameters(args):
    return {"data": str(args.dataset.resolve()), "epochs": args.epochs, "batch": args.batch, "imgsz": 320,
            "device": 0, "workers": 0, "seed": SEED, "amp": False, "plots": False,
            "project": str(PROJECT), "name": args.name, "exist_ok": False,
            "pretrained": True, "deterministic": True, "cache": False, "val": True,
            "save": True, "save_period": -1, "optimizer": "AdamW", "lr0": .001,
            "weight_decay": .0005, "cos_lr": True, "patience": args.epochs,
            "single_cls": True, "close_mosaic": 10, "mosaic": .7, "mixup": 0., "copy_paste": 0.,
            "verbose": False}


def build_protocol(args):
    before = tools.base_snapshot()
    if not tools.REPORT.exists():
        raise ValueError("Run setup_yolo_tools.py first")
    toolchain = json.loads(tools.REPORT.read_text())
    if toolchain.get("check", {}).get("passed") is not True:
        raise ValueError("Isolated toolchain has not passed its import/offline checks")
    if not tools.WEIGHTS.exists() or tools.digest(tools.WEIGHTS) != tools.ASSET_SHA256:
        raise ValueError("Official pinned YOLOv8n weights are missing or changed")
    dataset, manifest = dataset_metadata(args.dataset)
    protocol = {"schema_version": 1, "architecture": "yolov8n", "task": "detect", "nc": 1,
                "dataset": dataset, "protected_base": before,
                "pretrained": {"path": str(tools.WEIGHTS), "sha256": tools.ASSET_SHA256,
                               "repository": "ultralytics/assets", "tag": tools.ASSET_TAG, "asset_id": tools.ASSET_ID},
                "toolchain_report_sha256": tools.digest(tools.REPORT),
                "training_source_sha256": tools.digest(__file__), "setup_source_sha256": tools.digest(tools.__file__),
                "wrapper_sha256": tools.digest(ROOT/"fast_leaf_detector.py"),
                "promotion_protocol_sha256": tools.digest(ROOT/"docs/POTATO_DETECTOR_PROTOCOL.md"),
                "evaluator_source_sha256": tools.digest(ROOT/"evaluate_potato_detector.py"),
                "train": train_parameters(args),
                "export": {"format": "onnx", "imgsz": 320, "opset": 17, "nms": False,
                           "dynamic": False, "simplify": False, "batch": 1, "device": "cpu", "quantize": 32},
                "expected_onnx_input": [1, 3, 320, 320], "expected_onnx_output": [1, 5, 2100],
                "export_parity": export_parity_protocol(manifest),
                "test_used": False, "user_images_used": False, "external_services": False, "deploy": False,
                "selection": "Ultralytics best checkpoint by validation detection fitness; no test or user-image tuning",
                "limits": ["Synthetic potato leaves/public backgrounds; no claim of field accuracy",
                           "Dataset QA approval is required before any training starts",
                           "Export compatibility does not authorize replacing the current detector"]}
    return protocol, manifest


def verify_readiness(protocol, manifest):
    import math
    root = Path(protocol["dataset"]["yaml"]).parent
    readiness_path = root/"training_readiness.json"
    readiness = json.loads(readiness_path.read_text())
    if readiness.get("can_train") is not True or readiness.get("manifest_sha256") != protocol["dataset"]["manifest_sha256"]:
        raise ValueError("Root QA has not approved this exact dataset manifest")
    for row in manifest:
        for field, checksum_field in (("image", "image_sha256"), ("label", "label_sha256")):
            path = _dataset_file(row[field], root)
            if tools.digest(path) != row[checksum_field]:
                raise ValueError("Dataset image/label differs from its QA manifest")
        label_path = _dataset_file(row["label"], root)
        for line in label_path.read_text().splitlines():
            fields = line.split()
            if len(fields) != 5 or fields[0] != "0":
                raise ValueError("YOLO labels must have one class0 xywh object per row")
            values = [float(value) for value in fields[1:]]
            x, y, w, h = values
            if (not all(math.isfinite(value) for value in values) or not 0 < w <= 1 or not 0 < h <= 1
                    or x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6):
                raise ValueError("YOLO normalized box is invalid")
    return {"path": str(readiness_path), "sha256": tools.digest(readiness_path), "can_train": True,
            "manifest_sha256": readiness["manifest_sha256"], "images_and_labels_verified": len(manifest)}


def raw_parity(expected, actual, max_box_error, max_score_error):
    import numpy as np
    expected, actual = np.asarray(expected), np.asarray(actual)
    shapes_equal = expected.shape == actual.shape and actual.ndim == 3 and actual.shape[0] == 1 and actual.shape[1] == 5 and actual.shape[2] > 0
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    box_error = float(np.abs(actual[:, :4]-expected[:, :4]).max()) if shapes_equal and finite else None
    score_error = float(np.abs(actual[:, 4:]-expected[:, 4:]).max()) if shapes_equal and finite else None
    return {"shape": list(actual.shape), "reference_shape": list(expected.shape), "shapes_equal": shapes_equal,
            "finite": finite, "max_box_absolute_error": box_error, "max_score_absolute_error": score_error,
            "passed": bool(shapes_equal and finite and box_error <= max_box_error and score_error <= max_score_error)}


def decoded_parity(expected, actual, minimum_iou):
    """Require a complete same-class match, independent of score sort order."""
    def iou(left, right):
        a, b = left["box"], right["box"]
        intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
        union = max(0., a[2]-a[0])*max(0., a[3]-a[1]) + max(0., b[2]-b[0])*max(0., b[3]-b[1]) - intersection
        return intersection/union if union > 0 else 0.
    counts_equal = len(expected) == len(actual)
    classes_equal = sorted(row["detector_class"] for row in expected) == sorted(row["detector_class"] for row in actual)
    overlap = [[iou(a, b) if a["detector_class"] == b["detector_class"] else 0. for b in actual] for a in expected]
    edges = [sorted((j for j, value in enumerate(values) if value >= minimum_iou), key=lambda j: (-values[j], j)) for values in overlap]
    assigned = {}
    def assign(index, seen):
        for other in edges[index]:
            if other in seen:
                continue
            seen.add(other)
            if other not in assigned or assign(assigned[other], seen):
                assigned[other] = index
                return True
        return False
    complete = counts_equal and classes_equal and all(assign(index, set()) for index in range(len(expected)))
    matches = [{"reference_index": index, "onnx_index": other, "iou": overlap[index][other]} for other, index in sorted(assigned.items())]
    return {"reference_count": len(expected), "onnx_count": len(actual), "counts_equal": counts_equal,
            "classes_equal": classes_equal, "one_to_one_matched": bool(complete), "matches": matches,
            "minimum_matched_iou": min((row["iou"] for row in matches), default=1.) if complete else None,
            "passed": bool(complete)}


def runtime_parity_record(expected, actual, transform, config):
    from fast_leaf_detector import decode_output
    raw = raw_parity(expected, actual, config["max_raw_box_absolute_error"], config["max_raw_score_absolute_error"])
    # Do not allow the decoder's normal nonfinite filtering to hide bad raw output.
    if not raw["shapes_equal"] or not raw["finite"]:
        return {"raw": raw, "decoded": None, "passed": False}
    decoded = decoded_parity(decode_output(expected, transform, **config["decode"]),
                             decode_output(actual, transform, **config["decode"]), config["min_matched_iou"])
    return {"raw": raw, "decoded": decoded, "passed": raw["passed"] and decoded["passed"]}


def export_check(model, exported, run, protocol):
    if tools.digest(ROOT/"fast_leaf_detector.py") != protocol["wrapper_sha256"]:
        raise RuntimeError("Runtime detector wrapper changed since protocol preparation")
    import numpy as np
    import onnx
    import torch
    from PIL import Image
    from fast_leaf_detector import FastLeafDetector, letterbox
    onnx.checker.check_model(str(exported))
    wrapper = FastLeafDetector(exported, threads=2)
    if wrapper.input_size != (320, 320) or len(wrapper.names) != 1:
        raise RuntimeError("ONNX wrapper interface differs from the one-class detector")
    session = wrapper.session
    if session.get_inputs()[0].shape != [1, 3, 320, 320] or session.get_outputs()[0].shape != [1, 5, 2100]:
        raise RuntimeError("Expected raw YOLO output [1,5,2100], without baked NMS")
    reference = model.model.cpu().float().eval()
    rng = np.random.default_rng(SEED)
    records = []
    with torch.inference_mode():
        for index in range(6):
            image = (np.zeros((1, 3, 320, 320), dtype=np.float32) if index == 0
                     else rng.random((1, 3, 320, 320), dtype=np.float32))
            expected = reference(torch.from_numpy(image))
            expected = expected[0] if isinstance(expected, tuple) else expected
            expected = expected.detach().cpu().numpy()
            actual = session.run(None, {session.get_inputs()[0].name: image})[0]
            records.append({"sample": index, **raw_parity(expected, actual, .01, 1e-4)})
        config = protocol["export_parity"]
        dataset_root = Path(protocol["dataset"]["yaml"]).parent
        manifest_path = Path(protocol["dataset"]["manifest"])
        if tools.digest(manifest_path) != protocol["dataset"]["manifest_sha256"]:
            raise RuntimeError("Dataset manifest changed before export validation")
        if export_parity_protocol(json.loads(manifest_path.read_text())) != config:
            raise RuntimeError("Export validation inputs or rules differ from the frozen protocol")
        validation_records = []
        for index, sample in enumerate(config["samples"]):
            path = _dataset_file(sample["image"], dataset_root)
            if tools.digest(path) != sample["image_sha256"]:
                raise RuntimeError("Validation image changed before export validation")
            with Image.open(path) as image:
                tensor, transform = letterbox(image, wrapper.input_size)
            expected = reference(torch.from_numpy(tensor))
            expected = expected[0] if isinstance(expected, tuple) else expected
            expected = expected.detach().cpu().numpy()
            actual = session.run([wrapper.output_name], {wrapper.input_name: tensor})[0]
            validation_records.append({"sample": index, **sample,
                                       "transform": vars(transform),
                                       **runtime_parity_record(expected, actual, transform, config)})
    result = {"passed": all(row["passed"] for row in records + validation_records), "opset": 17, "nms": False, "nc": 1,
              "input": [1, 3, 320, 320], "output": [1, 5, 2100], "model_sha256": tools.digest(exported),
              "bytes": exported.stat().st_size, "metadata": wrapper.metadata, "samples": records,
              "validation_protocol": config, "validation_samples": validation_records,
              "wrapper_sha256": tools.digest(ROOT/"fast_leaf_detector.py"),
              "note": "Numerical/format and decoded runtime equivalence only; no test images or deployment."}
    tools.save(run/"export_validation.json", result)
    if not result["passed"]:
        raise RuntimeError("ONNX numerical equivalence failed; do not promote")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--name", default="potato_yolov8n_v1")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--train", action="store_true", help="Requires approved dataset QA and an unchanged prepared protocol")
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 100 or not 1 <= args.batch <= 64 or not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.name):
        parser.error("Invalid fixed experiment parameters")
    try:
        protocol, manifest = build_protocol(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    if not args.train:
        tools.save(args.protocol, {"prepared_at_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol})
        print(json.dumps({"prepared": True, "trained": False, "protocol": str(args.protocol),
                          "dataset_qa_required": True, "parameters": protocol["train"]}), flush=True)
        return 0
    try:
        prepared = json.loads(args.protocol.read_text())
    except (OSError, ValueError):
        parser.error("Prepare and review the training protocol before --train")
    if prepared.get("protocol") != protocol:
        parser.error("Protocol changed since preparation; review and prepare it again before training")
    try:
        readiness = verify_readiness(protocol, manifest)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    run = PROJECT/args.name
    if run.exists():
        parser.error("Candidate run already exists; choose another name and prepare a new protocol")
    before, _ = tools.activate()
    started = time.monotonic()
    with tools.offline_network() as blocked:
        import torch
        from ultralytics import YOLO
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; no automatic CPU fallback")
        torch.set_num_threads(3)
        model = YOLO(str(tools.WEIGHTS), task="detect")
        def freeze_before_train(trainer):
            tools.save(Path(trainer.save_dir)/"frozen_protocol.json", {"protocol": protocol, "qa": readiness,
                        "started_at_utc": datetime.now(timezone.utc).isoformat(), "gpu": torch.cuda.get_device_name(0)})
        model.add_callback("on_pretrain_routine_start", freeze_before_train)
        model.train(**protocol["train"])
        best = run/"weights/best.pt"
        if not best.exists():
            raise RuntimeError("Training did not produce the selected validation checkpoint")
        selected = YOLO(str(best), task="detect")
        exported = Path(selected.export(**protocol["export"]))
        validation = export_check(selected, exported, run, protocol)
    if tools.base_snapshot() != before:
        raise RuntimeError("Protected base distribution changed")
    result = {"trained": True, "deployed": False, "test_used": False, "user_images_used": False,
              "run": str(run), "best": str(best), "best_sha256": tools.digest(best),
              "exported": str(exported), "export_validation": validation,
              "elapsed_seconds": round(time.monotonic()-started, 3), "network_attempts_blocked": len(blocked)}
    tools.save(run/"training_summary.json", result)
    print(json.dumps({key: result[key] for key in ("trained", "deployed", "run", "best", "exported", "elapsed_seconds", "network_attempts_blocked")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
