"""CPU ONNX leaf/no-leaf verification using a frozen validation-only policy.

Scores are uncalibrated model outputs. Acceptance does not verify potato species
or disease. The caller owns localization, square crop padding and freshness.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


def verifier_input(image, config):
    """Match torchvision's PIL Resize(short edge) and CenterCrop exactly."""
    image = image.convert("RGB")
    side, edge = int(config["resize_short_edge"]), int(config["center_crop"])
    width, height = image.size
    if min(width, height) <= 0:
        raise ValueError("Empty verifier image")
    if width <= height:
        size = (side, int(side * height / width))
    else:
        size = (int(side * width / height), side)
    if image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    left, top = round((image.width - edge) / 2), round((image.height - edge) / 2)
    pixels = np.asarray(image.crop((left, top, left + edge, top + edge)), dtype=np.float32) / 255
    pixels = (pixels - np.asarray(config["mean"], dtype=np.float32)) / np.asarray(config["std"], dtype=np.float32)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])


class LeafVerifier:
    def __init__(self, model_path=None, config=None, threads=1, session=None):
        if config is None:
            if model_path is None:
                raise ValueError("A configuration or model path is required")
            config = Path(model_path).with_name("config.json")
        self.config = dict(config) if isinstance(config, dict) else json.loads(Path(config).read_text(encoding="utf-8"))
        if self.config.get("classes") != ["no_leaf", "leaf"]:
            raise ValueError("Expected classes ['no_leaf', 'leaf'] in that order")
        self.threshold = float(self.config["threshold"])
        edge, side = int(self.config["center_crop"]), int(self.config["resize_short_edge"])
        mean, std = np.asarray(self.config["mean"]), np.asarray(self.config["std"])
        if (not 0 < self.threshold <= 1 or edge != 224 or side < edge
                or mean.shape != (3,) or std.shape != (3,)
                or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
            raise ValueError("Invalid verifier preprocessing or threshold")
        if session is None:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = max(1, int(threads)), 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.session = session
        inputs = session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, edge, edge] or inputs[0].type != "tensor(float)":
            raise ValueError("Expected one fixed float32 NCHW RGB verifier input")
        self.input_name = inputs[0].name

    def verify(self, image):
        started = time.perf_counter()
        tensor = verifier_input(image, self.config)
        logits = np.asarray(self.session.run(None, {self.input_name: tensor})[0])
        if logits.shape != (1, 2) or not np.isfinite(logits).all():
            raise ValueError("Expected two finite verifier logits")
        values = np.exp(logits[0] - logits[0].max())
        score = float(values[1] / values.sum())
        return {"leaf_probability": score, "accepted": score >= self.threshold,
                "threshold": self.threshold, "ms": (time.perf_counter() - started) * 1000}
