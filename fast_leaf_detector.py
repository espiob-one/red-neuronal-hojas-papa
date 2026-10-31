"""CPU ONNX leaf localization only; disease classification is separate.

The bundled model was trained on tomato leaves. Its recall on potato plants or
moving rover footage must be measured before treating it as a field detector.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Letterbox:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    left: int
    top: int


def letterbox(image: Image.Image, size: tuple[int, int]) -> tuple[np.ndarray, Letterbox]:
    """Return normalized RGB NCHW and the exact rounded resize transform.

    ``size`` is (width, height). Only padding is added; the aspect ratio remains
    unchanged apart from unavoidable integer pixel rounding.
    """
    image = image.convert("RGB")
    width, height = image.size
    target_width, target_height = size
    if min(width, height, target_width, target_height) <= 0:
        raise ValueError("Image and model dimensions must be positive")
    ratio = min(target_width / width, target_height / height)
    resized_width = max(1, min(target_width, round(width * ratio)))
    resized_height = max(1, min(target_height, round(height * ratio)))
    left = (target_width - resized_width) // 2
    top = (target_height - resized_height) // 2
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_height, left:left + resized_width] = np.asarray(resized)
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= np.float32(255.0)
    transform = Letterbox(width, height, resized_width, resized_height, left, top)
    return tensor, transform


def restore_boxes(boxes: np.ndarray, transform: Letterbox) -> np.ndarray:
    """Map padded model xyxy boxes back to clamped original pixel coordinates."""
    restored = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    restored[:, [0, 2]] = (restored[:, [0, 2]] - transform.left) * (
        transform.original_width / transform.resized_width
    )
    restored[:, [1, 3]] = (restored[:, [1, 3]] - transform.top) * (
        transform.original_height / transform.resized_height
    )
    restored[:, [0, 2]] = restored[:, [0, 2]].clip(0, transform.original_width)
    restored[:, [1, 3]] = restored[:, [1, 3]].clip(0, transform.original_height)
    return restored


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45,
        limit: int = 8) -> list[int]:
    """Class-agnostic greedy NMS on xyxy boxes, with continuous pixel area."""
    if len(boxes) == 0 or limit <= 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores, kind="stable")
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0, boxes[:, 3] - boxes[:, 1]
    )
    kept = []
    while len(order) and len(kept) < limit:
        current = int(order[0])
        kept.append(current)
        rest = order[1:]
        if not len(rest):
            break
        upper_left = np.maximum(boxes[current, :2], boxes[rest, :2])
        lower_right = np.minimum(boxes[current, 2:], boxes[rest, 2:])
        lengths = np.maximum(0, lower_right - upper_left)
        intersection = lengths[:, 0] * lengths[:, 1]
        union = areas[current] + areas[rest] - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection),
                            where=union > 0)
        order = rest[overlap <= iou_threshold]
    return kept


def decode_output(output: np.ndarray, transform: Letterbox, threshold: float = 0.45,
                  iou_threshold: float = 0.45, limit: int = 8,
                  min_side: float = 8.0) -> list[dict]:
    """Decode standard YOLOv8 [1,4+nc,anchors] output; no disease labels."""
    values = np.asarray(output)
    if values.ndim != 3 or values.shape[0] != 1:
        raise ValueError(f"Expected [1, 4+classes, anchors], got {values.shape}")
    predictions = values[0].T
    if predictions.shape[1] < 5:
        raise ValueError(f"YOLOv8 output has no class scores: {values.shape}")
    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(predictions)), class_ids]
    valid = np.isfinite(predictions).all(axis=1) & (scores >= threshold)
    predictions, scores, class_ids = predictions[valid], scores[valid], class_ids[valid]
    if not len(predictions):
        return []
    centers = predictions[:, :2]
    halves = predictions[:, 2:4] / 2
    boxes = restore_boxes(np.concatenate((centers - halves, centers + halves), axis=1),
                          transform)
    valid = ((boxes[:, 2] - boxes[:, 0]) >= min_side) & (
        (boxes[:, 3] - boxes[:, 1]) >= min_side
    )
    boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]
    return [{"box": [float(value) for value in boxes[index]],
             "score": float(scores[index]), "detector_class": int(class_ids[index])}
            for index in nms(boxes, scores, iou_threshold, limit)]


class FastLeafDetector:
    """Load once, then call ``detect(PIL_RGB_image)`` on each sampled frame."""

    def __init__(self, path: str | Path, threads: int = 2):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        shape = model_input.shape
        if (len(shape) != 4 or shape[:2] != [1, 3] or
                not all(isinstance(value, int) and value > 0 for value in shape[2:])):
            raise ValueError(f"Expected fixed RGB NCHW [1,3,H,W], got {shape}")
        if model_input.type != "tensor(float)":
            raise ValueError(f"Expected float32 input, got {model_input.type}")
        self.input_name = model_input.name
        self.input_size = (shape[3], shape[2])
        self.output_name = self.session.get_outputs()[0].name
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        try:
            names = ast.literal_eval(self.metadata.get("names", "{}"))
            self.names = names if isinstance(names, (dict, list)) else {}
        except (SyntaxError, ValueError):
            self.names = {}
        self.threshold = 0.45
        self.iou_threshold = 0.45
        self.limit = 8

    def detect(self, image: Image.Image) -> list[dict]:
        tensor, transform = letterbox(image, self.input_size)
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        return decode_output(output, transform, self.threshold, self.iou_threshold, self.limit)
