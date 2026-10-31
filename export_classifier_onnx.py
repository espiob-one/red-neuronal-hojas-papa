"""Export the recorded MobileNetV2 classifier and verify ONNX equivalence.

Validation here checks numerical equivalence, not recognition accuracy in the
field. The existing training run, class order and preprocessing are preserved.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/20260911_213926")
    parser.add_argument("--output", type=Path, default=ROOT / "models/potato-mobilenet-v2-onnx")
    parser.add_argument("--per-class", type=int, default=9)
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class must be positive")

    config_path = args.run / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["architecture"] != "mobilenet_v2" or config["center_crop"] != 224:
        parser.error("Expected the recorded MobileNetV2 with 224-pixel input")
    torch.set_num_threads(4)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(config["classes"]))
    model.load_state_dict(torch.load(args.run / "best_weights.pt", map_location="cpu", weights_only=True))
    model.eval().cpu()

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "model.onnx"
    temporary_path = args.output / "model.pending.onnx"
    torch.onnx.export(
        model, torch.zeros(1, 3, 224, 224), temporary_path,
        input_names=["image"], output_names=["logits"],
        opset_version=17, dynamo=False,
        do_constant_folding=True,
    )
    onnx.checker.check_model(str(temporary_path))

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(temporary_path), sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape != [1, 3, 224, 224]:
        raise RuntimeError("Unexpected ONNX input shape")

    preprocess = transforms.Compose([
        transforms.Resize(config["resize_short_edge"]),
        transforms.CenterCrop(config["center_crop"]),
        transforms.ToTensor(),
        transforms.Normalize(config["mean"], config["std"]),
    ])
    manifest = json.loads((ROOT / "data/metadata/manifest.json").read_text(encoding="utf-8"))
    samples = []
    for label in config["classes"]:
        candidates = sorted(
            (row for row in manifest if row["label"] == label and row["split"] == "val"),
            key=lambda row: row["path"],
        )[:args.per_class]
        if len(candidates) != args.per_class:
            raise RuntimeError(f"Not enough validation images for {label}")
        samples.extend(candidates)

    comparisons = []
    with torch.inference_mode():
        for row in samples:
            with Image.open(ROOT / row["path"]) as image:
                tensor = preprocess(image.convert("RGB")).unsqueeze(0)
            reference = model(tensor).numpy()
            exported = session.run(["logits"], {"image": tensor.numpy()})[0]
            difference = float(np.max(np.abs(reference - exported)))
            if not np.isfinite(exported).all():
                raise RuntimeError(f"Non-finite ONNX output: {row['path']}")
            reference_index = int(reference.argmax(axis=1)[0])
            exported_index = int(exported.argmax(axis=1)[0])
            comparisons.append({
                "path": row["path"], "label": row["label"], "split": row["split"],
                "max_absolute_logit_difference": difference,
                "torch_class": config["classes"][reference_index],
                "onnx_class": config["classes"][exported_index],
                "argmax_match": reference_index == exported_index,
            })

    maximum_difference = max(row["max_absolute_logit_difference"] for row in comparisons)
    all_match = all(row["argmax_match"] for row in comparisons)
    tolerance = 1e-4
    passed = maximum_difference <= tolerance and all_match
    validation = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(args.run.resolve()),
        "source_weights": "best_weights.pt",
        "architecture": config["architecture"], "classes": config["classes"],
        "input_name": "image", "input_shape": [1, 3, 224, 224],
        "input_dtype": "float32", "output_name": "logits", "opset": 17,
        "provider": "CPUExecutionProvider",
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
        "selection": "First paths sorted per class in the existing validation split; no model selection or retraining",
        "images_per_class": args.per_class, "image_count": len(comparisons),
        "max_absolute_logit_difference": maximum_difference,
        "all_argmax_match": all_match, "absolute_logit_tolerance": tolerance,
        "passed": passed, "comparisons": comparisons,
        "limitation": "Numerical export equivalence only; this does not measure field accuracy or Raspberry Pi latency.",
    }
    (args.output / "export_validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError(f"ONNX equivalence failed: max difference {maximum_difference}, argmax match {all_match}")
    temporary_path.replace(model_path)
    shutil.copyfile(config_path, args.output / "config.json")
    print(json.dumps({
        "model": str(model_path.resolve()), "bytes": model_path.stat().st_size,
        "image_count": len(comparisons), "max_absolute_logit_difference": maximum_difference,
        "all_argmax_match": all_match, "passed": passed,
    }, indent=2))


if __name__ == "__main__":
    main()
