"""Atomic local collection of user-labelled Pi JPEGs, never automatic training.

Create one store per server. Completed records are UUID directories containing
image.jpg and metadata.json; .incomplete-* directories never count as captures.
The Pi system clock supplies the timestamp and may not yet be synchronized.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid

from PIL import Image, ImageStat

LABELS = frozenset(("Potato___Early_blight", "Potato___Late_blight", "Potato___healthy", "no_leaf"))
SOURCE_KINDS = frozenset(("real", "printed", "scene"))
MAX_JPEG_BYTES = 8 * 1024 * 1024
MIN_FREE_BYTES = 256 * 1024 * 1024
MAX_DIMENSION = 4096
MAX_PIXELS = 8_000_000
_SPECIMEN = re.compile(r"[A-Za-z0-9_-]{1,48}\Z", re.ASCII)
_CAPTURE = re.compile(r"[0-9a-f]{32}\Z", re.ASCII)
_SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


class CaptureError(Exception):
    """Safe human-readable failure; status can be used as an HTTP status."""
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


def _validate_fields(label, source_kind, specimen_id, frame_sequence, frame_age_seconds):
    # All caller fields are checked before reading any image or filesystem file.
    if not isinstance(label, str) or label not in LABELS:
        raise CaptureError("Elige una etiqueta válida para la captura.")
    if not isinstance(source_kind, str) or source_kind not in SOURCE_KINDS:
        raise CaptureError("Elige hoja real, impreso o escena sin hoja.")
    if (label == "no_leaf") != (source_kind == "scene"):
        raise CaptureError("La etiqueta sin hoja corresponde únicamente a una escena sin hoja.")
    if not isinstance(specimen_id, str) or not _SPECIMEN.fullmatch(specimen_id):
        raise CaptureError("El identificador admite de 1 a 48 letras, números, guiones o guiones bajos.")
    if type(frame_sequence) is not int or frame_sequence < 0:
        raise CaptureError("La cámara todavía no tiene un cuadro válido.")
    if (type(frame_age_seconds) not in (int, float) or not math.isfinite(frame_age_seconds)
            or not 0 <= frame_age_seconds <= 1.5):
        raise CaptureError("El cuadro es antiguo. Espera una imagen reciente de la cámara.")


def _validate_jpeg(jpeg):
    if not isinstance(jpeg, bytes) or not 0 < len(jpeg) <= MAX_JPEG_BYTES:
        raise CaptureError("La captura debe ser una imagen JPEG de hasta 8 MB.")
    try:
        with Image.open(BytesIO(jpeg)) as image:
            width, height = image.size
            if (image.format != "JPEG" or not 1 <= width <= MAX_DIMENSION
                    or not 1 <= height <= MAX_DIMENSION or width * height > MAX_PIXELS):
                raise CaptureError("La captura JPEG tiene un tamaño no admitido.")
            image.load()
            luminance = float(ImageStat.Stat(image.convert("L")).mean[0])
    except CaptureError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError):
        raise CaptureError("No se pudo leer la imagen JPEG de la cámara.") from None
    if luminance < 8:
        raise CaptureError("La imagen está demasiado oscura. Enciende una luz antes de guardar.")
    return width, height, luminance


class TrainingCaptureStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self._lock = threading.Lock()
        self._known = None
        self._count = 0
        self._last_success = None

    def _inventory(self):
        known, count = {}, 0
        for path in self.directory.iterdir():
            if not _CAPTURE.fullmatch(path.name) or not path.is_dir():
                continue
            data = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            if (not isinstance(data, dict) or data.get("capture_id") != path.name
                    or not isinstance(data.get("sha256"), str) or not _SHA.fullmatch(data["sha256"])
                    or data.get("label") not in LABELS or not (path / "image.jpg").is_file()):
                raise ValueError("Incomplete completed record")
            known[data["sha256"]] = data["label"]
            count += 1
        self._known, self._count = known, count

    @staticmethod
    def _write(path, content):
        with path.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())

    def save(self, jpeg: bytes, *, label, source_kind, specimen_id,
             frame_sequence, frame_age_seconds):
        _validate_fields(label, source_kind, specimen_id, frame_sequence, frame_age_seconds)
        width, height, luminance = _validate_jpeg(jpeg)
        sha = hashlib.sha256(jpeg).hexdigest()
        with self._lock:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                if self._known is None:
                    self._inventory()
                # Duplicates, including conflicting labels, always return409.
                if sha in self._known:
                    message = ("Esta imagen ya está guardada." if self._known[sha] == label else
                               "Esta imagen ya está guardada con otra etiqueta. Revisa el registro antes de cambiarla.")
                    raise CaptureError(message, 409)
                now = time.monotonic()
                if self._last_success is not None and now - self._last_success < 1:
                    raise CaptureError("Espera un segundo entre capturas.", 429)
                if shutil.disk_usage(self.directory).free < MIN_FREE_BYTES:
                    raise CaptureError("No queda espacio suficiente para guardar otra captura.", 507)
                capture_id = uuid.uuid4().hex
                destination = self.directory / capture_id
                staging = self.directory / (".incomplete-" + capture_id)
                staging.mkdir(mode=0o700, exist_ok=False)
                metadata = {
                    "schema_version": 1, "capture_id": capture_id, "label": label,
                    "source_kind": source_kind, "specimen_id": specimen_id,
                    "group_id": source_kind + ":" + specimen_id, "sha256": sha,
                    "image_path": "image.jpg", "bytes": len(jpeg), "width": width, "height": height,
                    "mean_luminance": luminance, "frame_sequence": frame_sequence,
                    "frame_age_seconds": frame_age_seconds,
                    "captured_utc": datetime.now(timezone.utc).isoformat(),
                    "clock_source": "Pi_system_clock", "clock_accuracy_confirmed": False,
                    "source": "Pi_camera", "label_source": "user_declaration",
                    "ground_truth_biologically_confirmed": False, "status": "pending_review",
                    "model_predictions_not_truth": True, "automatic_training_authorized": False,
                }
                self._write(staging / "image.jpg", jpeg)
                self._write(staging / "metadata.json",
                            (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
                if destination.exists():
                    raise OSError("Capture identifier collision")
                staging.rename(destination)
                self._known[sha] = label
                self._count += 1
                self._last_success = now
                return {"capture_id": capture_id, "label": label, "source_kind": source_kind,
                        "specimen_id": specimen_id, "sha256": sha,
                        "frame_sequence": frame_sequence, "count": self._count}
            except CaptureError:
                raise
            except (OSError, ValueError, TypeError):
                # No pathname, decoder detail or exception text reaches the viewer.
                # Any failed staging directory remains explicitly incomplete.
                raise CaptureError("No se pudo guardar la captura. Revisa el almacenamiento de la Raspberry.", 500) from None
