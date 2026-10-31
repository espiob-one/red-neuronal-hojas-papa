"""Export the explicitly scoped known-print demo; never train or enable field use."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch

import train_robust_classifier as base
from rover_motion import classifier_input, classify_scores

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "docs/planes/IMPRESOS_MAQUETA_20261026.md"
REPORT = ROOT / "reports/print_demo_20261026"
OUTPUT = ROOT / "models/printed-demo-v1"
CHECKPOINT = ROOT / "runs/print_head_20261026_201041/epoch_8_weights.pt"
CHECKPOINT_SHA = "1fac56ffcb27b7fe9b931ec5ee817132d15581a6075dbb3c3c38d878e5a7b1cc"
PRINTS = ROOT / "data/user_prints_20261026/training_manifest.json"
PRINTS_SHA = "cb9b70b54b2194d05f62fdff4c4295b8a1ce9332ec429dcf761bb82ef4e35cee"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def decision(logits, classes):
    result = classify_scores(logits, classes)
    return int(np.argmax(logits[0])), result["class"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true", help="Run only after the new manual-demo protocol is frozen")
    args = parser.parse_args(argv)
    if not args.export:
        parser.error("Use --export after the explicit manual-demo protocol is frozen")
    if not PROTOCOL.is_file():
        raise FileNotFoundError("Manual-demo protocol must exist before export")
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("Never overwrite a previous demo export or report")
    if sha(CHECKPOINT) != CHECKPOINT_SHA or sha(PRINTS) != PRINTS_SHA:
        raise RuntimeError("Frozen checkpoint or development manifest changed")
    old_report_path = ROOT / "reports/print_head_20261026/report.json"
    old_report = read(old_report_path)
    if old_report["status"] != "REJECTED" or old_report["selected_checkpoint_sha256"] != CHECKPOINT_SHA:
        raise RuntimeError("Preserve the original general-scope rejection")
    general_path = ROOT / "models/potato-robust-candidate/config.json"
    general = read(general_path)
    classes = general["classes"]
    rows = read(PRINTS)["images"]
    if len(rows) != 37 or any(r["split"] != "train_development" for r in rows):
        raise RuntimeError("The37 fixed print rows remain development training only")
    manifest_path = ROOT / "data/metadata/manifest.json"
    public = read(manifest_path)
    controls = []
    for label in classes:
        controls.extend(sorted((r for r in public if r["split"] == "val" and r["label"] == label),
                               key=lambda r:r["path"])[:8])
    if len(controls) != 24:
        raise RuntimeError("Require exactly8 reused clean val controls per class")
    inputs = [("printed_train_development", r) for r in rows] + [("reused_public_val_export_only", r) for r in controls]
    protected = {str(p):sha(p) for p in (PROTOCOL, CHECKPOINT, PRINTS, old_report_path,
                                        general_path, manifest_path, Path(__file__))}
    for _, row in inputs:
        path = ROOT / row["path"]
        if sha(path) != row["sha256"]:
            raise RuntimeError("Input image hash changed")
        protected[str(path)] = row["sha256"]
    config = {k:general[k] for k in ("classes", "mean", "std", "resize_short_edge", "center_crop",
                                    "crop_margin", "square_pad", "square_pad_color")}
    config.update(architecture="mobilenet_v2", scope="known_printed_demo", status="PENDING_EXPORT_PARITY",
                  display_name="Maqueta · impresos conocidos", source_checkpoint=str(CHECKPOINT.relative_to(ROOT)),
                  source_checkpoint_sha256=CHECKPOINT_SHA, general_validation_status="REJECTED",
                  general_scope_eligible=False, independent_validation_available=False,
                  print_manifest_sha256=PRINTS_SHA, user_data_role="37 training/development crops; repeated known designs",
                  printed_source_identity_unknown=True, decision_policy={"softmax":.7,"margin":.2},
                  output="3 logits; scores are not calibrated disease probabilities", deployment_authorized=False,
                  limitations=["Experimental manual mode for known printed mockup leaves only",
                               "Training fit is not independent accuracy or field performance",
                               "The separate original classifier remains the general mode",
                               "Darkness and capture quality guards remain unchanged"])
    REPORT.mkdir(parents=True, exist_ok=False)
    staging = REPORT / "incomplete-export"
    staging.mkdir()
    write(REPORT / "frozen_inputs.json", {"sha256":protected,
          "classifier_input_source_sha256":hashlib.sha256(inspect.getsource(classifier_input).encode()).hexdigest(),
          "classify_scores_source_sha256":hashlib.sha256(inspect.getsource(classify_scores).encode()).hexdigest(),
          "inputs":[{"cohort":cohort,"path":row["path"],"sha256":row["sha256"]} for cohort,row in inputs]})
    write(staging / "config.json", config)
    try:
        torch.set_num_threads(3)
        model = base.create_model(classes, CHECKPOINT).eval().cpu()
        model_path = staging / "model.onnx"
        torch.onnx.export(model, torch.zeros(1,3,224,224), str(model_path), input_names=["image"],
                          output_names=["logits"], opset_version=17, dynamo=False, do_constant_folding=True)
        graph = onnx.load(str(model_path))
        onnx.helper.set_model_props(graph, {"scope":"known_printed_demo", "status":"EXPERIMENTAL_KNOWN_PRINTS",
                                            "source_checkpoint_sha256":CHECKPOINT_SHA})
        onnx.checker.check_model(graph)
        onnx.save(graph, str(model_path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        if session.get_inputs()[0].shape != [1,3,224,224] or session.get_outputs()[0].shape != [1,3]:
            raise RuntimeError("Unexpected fixed classifier interface")
        comparisons = []
        with torch.inference_mode():
            for cohort, row in inputs:
                with Image.open(ROOT / row["path"]) as image:
                    tensor = classifier_input(image.convert("RGB"), config)
                expected = model(torch.from_numpy(tensor)).numpy()
                actual = session.run(None, {"image":tensor})[0]
                error = float(np.abs(expected-actual).max())
                matches = decision(expected,classes) == decision(actual,classes)
                comparisons.append({"cohort":cohort,"path":row["path"],"sha256":row["sha256"],
                                    "max_absolute_logit_difference":error,"argmax_and_abstention_match":matches})
        parity = {"passed":all(c["max_absolute_logit_difference"] <= 1e-4 and c["argmax_and_abstention_match"] for c in comparisons),
                  "inputs":len(comparisons),"printed_development":37,"reused_clean_public_val":24,
                  "max_absolute_logit_difference":max(c["max_absolute_logit_difference"] for c in comparisons),
                  "all_argmax_and_abstention_match":all(c["argmax_and_abstention_match"] for c in comparisons),
                  "policy":{"softmax":.7,"margin":.2},"purpose":"Numerical export parity ONLY; no performance selection", "comparisons":comparisons}
        write(REPORT / "export_validation.json", parity)
        write(staging / "export_validation.json", parity)
        if not parity["passed"] or any(sha(Path(p)) != value for p,value in protected.items()):
            raise RuntimeError("Export parity or frozen-input integrity failed")
        config.update(status="EXPERIMENTAL_KNOWN_PRINTS", onnx_sha256=sha(model_path))
        write(staging / "config.json", config)
        if OUTPUT.exists():
            raise FileExistsError("Never overwrite existing model")
        staging.rename(OUTPUT)
    except Exception as exc:
        config["status"] = "REJECTED_EXPORT_PARITY"
        if staging.exists():
            write(staging / "config.json", config)
        write(REPORT / "report.json", {"status":config["status"],"error_type":type(exc).__name__,
                                      "output_published":False,"general_validation_status":"REJECTED"})
        raise
    report = {"created_utc":datetime.now(timezone.utc).isoformat(),"status":config["status"],
              "scope":config["scope"],"onnx_sha256":config["onnx_sha256"],"config_sha256":sha(OUTPUT/"config.json"),
              "checkpoint_sha256":CHECKPOINT_SHA,"output":str(OUTPUT),"parity":parity,
              "general_validation_status":"REJECTED","independent_accuracy_claim":False,
              "training_performed":False,"pi_accessed":False,"deployed":False}
    write(REPORT / "report.json", report)
    print(json.dumps({k:report[k] for k in ("status","scope","onnx_sha256","output","deployed")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
