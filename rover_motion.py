"""Full-frame leaf detection, bounded tracking and independent crop classification."""
import argparse
import base64
from collections import deque
import io
import json
from pathlib import Path
import signal
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen
from urllib.parse import parse_qs

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from fast_leaf_detector import FastLeafDetector
from leaf_tracks import LeafTracker
from motion_tracking_candidate import MotionLeafTracker
from viewer_network import address, client_allowed, host_allowed, same_origin

LABELS = {"Potato___Early_blight": "Tizón temprano", "Potato___Late_blight": "Tizón tardío", "Potato___healthy": "Sana"}
RESULT_MAX_AGE_SECONDS = 2.0
REJECT_RETRY_SECONDS = 1.0


def leaf_crop(image, box, padding=0.12):
    """Keep the complete detected box, with context, rather than image center."""
    x1, y1, x2, y2 = box
    dx, dy = (x2-x1)*padding, (y2-y1)*padding
    bounds = (max(0, int(x1-dx)), max(0, int(y1-dy)),
              min(image.width, int(np.ceil(x2+dx))), min(image.height, int(np.ceil(y2+dy))))
    crop = image.crop(bounds).convert("RGB")
    return ImageOps.pad(crop, (max(crop.size),)*2, color=(128, 128, 128))


def sharpness(image):
    gray = np.asarray(image.convert("L").resize((160, 160)), dtype=np.float32)
    lap = -4*gray[1:-1, 1:-1]+gray[:-2, 1:-1]+gray[2:, 1:-1]+gray[1:-1, :-2]+gray[1:-1, 2:]
    return float(lap.var())


def capture_ready(image, box, crop):
    x1, y1, x2, y2 = box
    return (min(x2-x1, y2-y1) >= 40 and sharpness(crop) >= 30
            and x1 >= 3 and y1 >= 3 and x2 <= image.width-3 and y2 <= image.height-3)


def appearance(crop):
    return np.asarray(crop.resize((16,16)), dtype=np.float32)/255.0


def appearance_distance(first,second):
    # A strong change in one color channel must not be diluted by unchanged
    # channels. This is a content association guard, not a disease feature.
    return float(np.abs(first-second).mean(axis=(0,1)).max())


def classifier_input(crop, config):
    side, edge = config["resize_short_edge"], config["center_crop"]
    crop = crop.resize((side, side), Image.Resampling.BILINEAR)
    offset = (side-edge)//2
    crop = crop.crop((offset, offset, offset+edge, offset+edge))
    array = np.asarray(crop, dtype=np.float32)/255.0
    array = (array-np.asarray(config["mean"], dtype=np.float32))/np.asarray(config["std"], dtype=np.float32)
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def classify_scores(logits, classes):
    exp = np.exp(logits[0]-np.max(logits[0]))
    scores = exp/exp.sum()
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    certain = float(scores[best]) >= .7 and float(scores[best]-scores[order[1]]) >= .2
    return {"label": ("Orientativo: " if certain else "No concluyente · mayor puntuación: ")+LABELS[classes[best]],
            "class": classes[best] if certain else None,
            "score": float(scores[best]) if certain else None,
            "scores": [{"label": LABELS[c], "class": c, "score": float(s)} for c, s in zip(classes, scores)]}


class State:
    def __init__(self):
        self.condition = threading.Condition(threading.RLock())
        self.stop = threading.Event()
        self.frame = None
        self.sequence = 0
        self.frame_time = 0
        self.camera_epoch = 0
        self.width = self.height = 0
        self.camera_error = None
        self.detector_error = self.verifier_error = self.classifier_error = None
        self.phase = "Iniciando detector"
        self.telemetry = None
        self.tracker = MotionLeafTracker()
        self.jobs = deque(maxlen=2)
        self.detector_ms = 0
        self.classifier_ms = 0
        self.detected_at = 0
        self.detection_frames = 0
        self.classification_count = 0
        self.capture_store = None
        self.capture_error = None
        self.analysis_mode = "general"
        self.available_modes = ("general",)
        self.last_prediction = None
        self.appearances = {}
        self.current_appearances = {}
        self.changed = {}
        self.rejections = {}
        self.verifier_ms = 0
        self.verifier_checks = 0
        self.verifier_rejections = 0

    @property
    def model_error(self):
        # Workers own their own failures. A healthy detector cannot hide a
        # failed classifier, and classifier recovery cannot hide detection loss.
        return self.detector_error or self.verifier_error or self.classifier_error

    def set_analysis_mode(self, mode):
        with self.condition:
            if not isinstance(mode, str) or mode not in self.available_modes:
                raise ValueError("Elige un modo de análisis disponible.")
            changed = mode != self.analysis_mode
            if changed:
                self.analysis_mode = mode
                self.camera_epoch += 1
                self.tracker.reset(time.monotonic())
                self.jobs.clear()
                self.appearances.clear()
                self.current_appearances.clear()
                self.changed.clear()
                self.rejections.clear()
                self.last_prediction = None
                self.detected_at = 0
                self.classifier_error = None
                self.classifier_ms = 0
                self.phase = "Buscando hojas"
                self.condition.notify_all()
            return {"mode": self.analysis_mode, "changed": changed}

    def status(self):
        with self.condition:
            now = time.monotonic()
            fresh = self.frame is not None and now-self.frame_time < 1.5 and not self.camera_error and not self.model_error
            tracks = self.tracker.snapshot(now)["tracks"] if fresh and now-self.detected_at < 1.2 else []
            visible=[]
            for track in tracks:
                if not track["observed"]:
                    continue
                tid=track["track_id"]
                current=self.current_appearances.get(tid)
                accepted=self.appearances.get(tid)
                same=(current is None or accepted is None or appearance_distance(current,accepted)<=.14)
                rejected=self.rejections.get(tid)
                if (rejected and current is not None and now-rejected["frame_time"]<REJECT_RETRY_SECONDS
                        and appearance_distance(current,rejected["appearance"])<=.14):
                    continue
                recent=(track["result"] is not None
                        and now-track["result"]["frame_time"]<=RESULT_MAX_AGE_SECONDS)
                visible.append({**track,"result":track["result"] if track["quality_ok"] and same and recent else None})
            tracks=visible
            prediction = max((t["result"] for t in tracks if t["result"] is not None),
                             key=lambda p:p["frame_time"], default=None)
            return {"version": "motion-v3.1.0", "frames": self.sequence, "frame_age_seconds": now-self.frame_time,
                    "camera_error": self.camera_error, "model_error": self.model_error,
                    "phase": "Buscando hojas" if self.phase == "Hojas en seguimiento" and not tracks else self.phase,
                    "device": "Raspberry CPU · ONNX", "width": self.width, "height": self.height,
                    "tracks": tracks, "detector_ms": self.detector_ms, "classifier_ms": self.classifier_ms,
                    "detection_frames": self.detection_frames, "classification_count": self.classification_count,
                    "verifier_ms": self.verifier_ms, "verifier_checks": self.verifier_checks,
                    "verifier_rejections": self.verifier_rejections,
                    "prediction": prediction,
                    "prediction_age_seconds": now-prediction["frame_time"] if prediction else None,
                    "training_capture_enabled": self.capture_store is not None,
                    "training_capture_error": self.capture_error,
                    "analysis_mode": self.analysis_mode, "available_modes": list(self.available_modes),
                    "telemetry": self.telemetry.snapshot() if self.telemetry else {}}


def read_camera(state, url):
    while not state.stop.is_set():
        try:
            with urlopen(url, timeout=2) as response:
                while not state.stop.is_set():
                    line = response.readline(4096)
                    if not line:
                        raise EOFError("Fin de video")
                    if line.strip() != b"--FRAME":
                        continue
                    length = 0
                    for _ in range(20):
                        header = response.readline(4096)
                        if header == b"\r\n":
                            break
                        if header.lower().startswith(b"content-length:"):
                            length = int(header.split(b":", 1)[1])
                    if not 0 < length <= 8_000_000:
                        raise ValueError("Cuadro inválido")
                    frame = response.read(length)
                    if len(frame) != length or not frame.endswith(b"\xff\xd9"):
                        raise ValueError("JPEG incompleto")
                    with state.condition:
                        state.frame, state.frame_time = frame, time.monotonic()
                        state.sequence += 1
                        state.camera_error = None
                        state.condition.notify_all()
        except Exception as exc:
            with state.condition:
                state.camera_error = str(exc)
                state.camera_epoch += 1
                state.tracker.reset(time.monotonic())
                state.jobs.clear()
                state.appearances.clear()
                state.current_appearances.clear()
                state.changed.clear()
                state.rejections.clear()
            state.stop.wait(1)


def detect_loop(state, detector):
    last = 0
    while not state.stop.is_set():
        with state.condition:
            state.condition.wait_for(lambda: state.sequence != last or state.stop.is_set(), timeout=1)
            if state.stop.is_set():
                return
            if state.frame is None or last == state.sequence or time.monotonic()-state.frame_time > 1.0:
                last = state.sequence
                continue
            frame, last, frame_time, epoch = state.frame, state.sequence, state.frame_time, state.camera_epoch
        started = time.monotonic()
        try:
            with Image.open(io.BytesIO(frame)) as source:
                image = source.convert("RGB")
            detections = detector.detect(image)
            for detection in detections:
                crop = leaf_crop(image, detection["box"])
                # Do not turn motion blur or a tiny patch into a diagnosis.
                detection["ready"] = capture_ready(image, detection["box"], crop)
            with state.condition:
                if epoch != state.camera_epoch or state.camera_error or time.monotonic()-frame_time > 1.2:
                    continue
                state.width, state.height = image.size
                state.detector_ms = round((time.monotonic()-started)*1000, 1)
                state.detected_at = time.monotonic()
                state.detection_frames += 1
                tracks = state.tracker.update(detections, time.monotonic())
                active_ids = {t["track_id"] for t in state.tracker.snapshot()["tracks"]}
                state.appearances = {k:v for k,v in state.appearances.items() if k in active_ids}
                state.current_appearances = {k:v for k,v in state.current_appearances.items() if k in active_ids}
                state.changed = {k:v for k,v in state.changed.items() if k in active_ids}
                state.rejections = {k:v for k,v in state.rejections.items() if k in active_ids}
                for track in tracks:
                    tid = track["track_id"]
                    crop = leaf_crop(image, track["box"])
                    state.current_appearances[tid] = appearance(crop)
                    prior = state.appearances.get(tid)
                    changed = (track["quality_ok"] and track["result"] is not None and prior is not None
                               and frame_time-track["result"]["frame_time"] > 1
                               and appearance_distance(appearance(crop),prior) > .14)
                    state.changed[tid] = state.changed.get(tid, 0)+1 if changed else 0
                    expired=(track["result"] is not None
                             and frame_time-track["result"]["frame_time"]>=RESULT_MAX_AGE_SECONDS)
                    if state.changed[tid] >= 2 or expired:
                        state.tracker.invalidate_classification(tid)
                        track["ready"] = track["confirmed"] and track["quality_ok"]
                        state.changed[tid] = 0
                    if not track["ready"] or not track["confirmed"]:
                        continue
                    rejection=state.rejections.get(tid)
                    if (rejection and frame_time-rejection["frame_time"]<REJECT_RETRY_SECONDS
                            and appearance_distance(state.current_appearances[tid],rejection["appearance"])<=.14):
                        continue
                    token = state.tracker.begin_classification(track["track_id"], track["last_seen"])
                    if token is None:
                        continue
                    if len(state.jobs) == state.jobs.maxlen:
                        removed = state.jobs.popleft()
                        state.tracker.cancel_classification(removed[0])
                    state.jobs.append((token, crop, frame_time, last, track["track_id"]))
                state.phase = "Hojas en seguimiento" if tracks else "Buscando hojas"
                state.detector_error = None
                state.condition.notify_all()
        except Exception as exc:
            with state.condition:
                state.detector_error = "Detector: "+str(exc)
        state.stop.wait(max(0, .1-(time.monotonic()-started)))


def classify_loop(state, session, config, verifier=None, printed_session=None, printed_config=None):
    input_name = session.get_inputs()[0].name
    printed_input = printed_session.get_inputs()[0].name if printed_session is not None else None
    while not state.stop.is_set():
        with state.condition:
            state.condition.wait_for(lambda: bool(state.jobs) or state.stop.is_set(), timeout=1)
            if state.stop.is_set():
                return
            if not state.jobs:
                continue
            token, crop, frame_time, sequence, track_id = state.jobs.popleft()
            if time.monotonic()-frame_time > 1.2 or state.camera_error:
                state.tracker.cancel_classification(token)
                continue
            mode = getattr(state, "analysis_mode", "general")
            epoch = state.camera_epoch
            if mode not in ("general", "printed") or (mode == "printed" and
                    (printed_session is None or printed_config is None)):
                state.classifier_error = "No está disponible el clasificador del modo seleccionado."
                state.tracker.cancel_classification(token)
                continue
            active_session, active_config, active_input = (
                (printed_session, printed_config, printed_input) if mode == "printed" else
                (session, config, input_name))
        started = time.monotonic()
        error_stage = "verifier" if verifier else "classifier"
        try:
            verification = verifier.verify(crop) if verifier else None
            if verification is not None:
                with state.condition:
                    if state.camera_epoch != epoch or getattr(state, "analysis_mode", "general") != mode:
                        state.tracker.cancel_classification(token)
                        continue
                    state.verifier_ms = round(verification["ms"], 1)
                    state.verifier_checks += 1
                    state.verifier_error = None
                if not verification["accepted"]:
                    with state.condition:
                        current=state.current_appearances.get(track_id)
                        same_content=current is not None and appearance_distance(current,appearance(crop))<=.14
                        cancelled=state.tracker.cancel_classification(token)
                        if (cancelled and same_content and not state.camera_error and state.camera_epoch == epoch
                                and getattr(state, "analysis_mode", "general") == mode):
                            state.rejections[track_id]={"frame_time":frame_time,"appearance":appearance(crop)}
                            state.verifier_rejections += 1
                    continue
            error_stage = "classifier"
            logits = active_session.run(None, {active_input: classifier_input(crop, active_config)})[0]
            result = classify_scores(logits, active_config["classes"])
            result["analysis_mode"] = mode
            if mode == "printed":
                result["label"] = "Maqueta · " + result["label"]
            output = io.BytesIO()
            crop.resize((224, 224)).save(output, "JPEG", quality=85)
            result.update({"crop": base64.b64encode(output.getvalue()).decode(), "frame_time": frame_time,
                           "sequence": sequence, "track_id": track_id,
                           "inference_ms": round((time.monotonic()-started)*1000, 1)})
            if verification is not None:
                result["leaf_score"]=verification["leaf_probability"]
            with state.condition:
                if state.camera_epoch != epoch or getattr(state, "analysis_mode", "general") != mode:
                    state.tracker.cancel_classification(token)
                    continue
                state.classifier_error = None
                state.classifier_ms = result["inference_ms"]
                current=state.current_appearances.get(track_id)
                same_content=current is not None and appearance_distance(current,appearance(crop))<=.14
                if not state.camera_error and same_content and state.tracker.finish_classification(token, result, time.monotonic()):
                    state.classification_count += 1
                    state.last_prediction = result
                    state.appearances[track_id] = appearance(crop)
                    state.rejections.pop(track_id,None)
                else:
                    state.tracker.cancel_classification(token)
        except Exception as exc:
            with state.condition:
                if state.camera_epoch != epoch or getattr(state, "analysis_mode", "general") != mode:
                    state.tracker.cancel_classification(token)
                    continue
                if error_stage == "verifier":
                    state.verifier_error = "Verificador: "+str(exc)
                else:
                    state.classifier_error = "Clasificador: "+str(exc)
                state.tracker.cancel_classification(token)


def make_handler(state, args, access=None):
    """Build the same HTTP handler used on the Pi, also available for tests."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_data(self, data, kind, status=200, headers=()):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            # Form POSTs under no-referrer send Origin:null and fail our
            # same-origin login check even when submitted from this viewer.
            self.send_header("Referrer-Policy", "same-origin")
            for name, value in headers:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        def allowed(self):
            return (client_allowed(self.client_address[0], args.allow_client, args.allow_subnet)
                    and host_allowed(self.headers.get("Host"), self.connection.getsockname()[0],
                                     args.port, args.hostnames.split(",")))

        def authenticated(self):
            return (access is None or address(self.client_address[0]).is_loopback
                    or access.authenticated(self.headers.get("Cookie")))

        def send_json(self, value, status=200):
            self.send_data(json.dumps(value, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8", status)

        def capture_example(self):
            # A browser POST must prove the same origin and hold the normal
            # viewer session. Reading the current camera happens only on click.
            if not self.headers.get("Origin"):
                self.send_json({"error": "Abre el visor y guarda el ejemplo desde ahí."}, 403)
                return
            if not self.authenticated():
                self.send_json({"error": "Entra al visor con tu código antes de guardar."}, 401)
                return
            self.connection.settimeout(5)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if (not 0 < length <= 2048 or self.headers.get_content_type() != "application/json"
                        or self.headers.get("Transfer-Encoding")):
                    self.send_json({"error": "Los datos del ejemplo no son válidos."}, 400)
                    return
                body = self.rfile.read(length)
                if len(body) != length:
                    self.send_json({"error": "La solicitud llegó incompleta."}, 400)
                    return
                values = json.loads(body.decode("utf-8"))
                if not isinstance(values, dict) or set(values) != {"label", "source_kind", "specimen_id"}:
                    self.send_json({"error": "Elige origen, clase e identificador de la hoja."}, 400)
                    return
            except (ValueError, UnicodeError, RecursionError):
                self.send_json({"error": "Los datos del ejemplo no son válidos."}, 400)
                return
            except (OSError, TimeoutError):
                self.send_json({"error": "No llegó la solicitud completa; vuelve a intentar."}, 408)
                return
            with state.condition:
                store = getattr(state, "capture_store", None)
                frame = state.frame
                sequence = state.sequence
                frame_time = getattr(state, "frame_time", 0)
                camera_error = getattr(state, "camera_error", None)
            if store is None:
                self.send_json({"error": "No está disponible el guardado de ejemplos."}, 503)
                return
            if not frame or camera_error:
                self.send_json({"error": "Espera a que la cámara muestre una imagen actual."}, 409)
                return
            from training_capture import CaptureError
            try:
                result = store.save(frame, label=values["label"], source_kind=values["source_kind"],
                                    specimen_id=values["specimen_id"], frame_sequence=sequence,
                                    frame_age_seconds=time.monotonic()-frame_time)
                self.send_json(result)
            except CaptureError as exc:
                self.send_json({"error": exc.message}, exc.status)
            except (OSError, ValueError):
                self.send_json({"error": "No se pudo guardar el ejemplo en la Raspberry."}, 503)

        def change_analysis_mode(self):
            if not self.headers.get("Origin"):
                self.send_json({"error": "Cambia de modo desde el visor."}, 403)
                return
            if not self.authenticated():
                self.send_json({"error": "Entra al visor con tu código antes de cambiar de modo."}, 401)
                return
            self.connection.settimeout(5)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if (not 0 < length <= 128 or self.headers.get_content_type() != "application/json"
                        or self.headers.get("Transfer-Encoding")):
                    raise ValueError("Los datos del modo no son válidos.")
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("La solicitud llegó incompleta.")
                values = json.loads(body.decode("utf-8"))
                if not isinstance(values, dict) or set(values) != {"mode"}:
                    raise ValueError("Elige un modo de análisis disponible.")
                self.send_json(state.set_analysis_mode(values["mode"]))
            except (ValueError, UnicodeError, RecursionError):
                self.send_json({"error": "Elige un modo de análisis disponible."}, 400)
            except (OSError, TimeoutError):
                self.send_json({"error": "No llegó la solicitud completa; vuelve a intentar."}, 408)

        def do_POST(self):
            if not self.allowed() or not same_origin(self.headers.get("Origin"), self.headers.get("Host")):
                self.send_error(403)
                return
            if self.path == "/api/training-capture":
                try:
                    self.capture_example()
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass
                return
            if self.path == "/api/analysis-mode":
                try:
                    self.change_analysis_mode()
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass
                return
            if self.path != "/login" or access is None:
                self.send_error(404)
                return
            self.connection.settimeout(5)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 or self.headers.get_content_type() != "application/x-www-form-urlencoded":
                    self.send_error(400)
                    return
                body = self.rfile.read(length)
                if len(body) != length:
                    self.send_error(400)
                    return
                values = parse_qs(body.decode("utf-8"), max_num_fields=4)
                codes = values.get("code", [])
                code = codes[0] if len(codes) == 1 else ""
                peer = self.client_address[0]
                if access.check_password(code, peer):
                    self.send_data(b"", "text/plain", 303,
                                   (("Location", "/"), ("Set-Cookie", access.issue_cookie())))
                else:
                    retry = access.retry_after(peer)
                    self.send_data(access.login_page(error=True), "text/html; charset=utf-8",
                                   429 if retry else 401, (("Retry-After", str(retry)),) if retry else ())
            except (ValueError, UnicodeError):
                self.send_error(400)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

        def do_GET(self):
            if not self.allowed():
                self.send_error(403)
                return
            self.connection.settimeout(5)
            try:
                if not self.authenticated():
                    self.send_data(access.login_page(), "text/html; charset=utf-8",
                                   200 if self.path.split("?")[0] == "/" else 401)
                    return
                if self.path.split("?")[0] == "/":
                    self.send_data(Path(__file__).with_name("rover_motion.html").read_bytes(), "text/html; charset=utf-8")
                elif self.path == "/status":
                    self.send_data(json.dumps(state.status()).encode(), "application/json")
                elif self.path.split("?")[0] == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    last = -1
                    while not state.stop.is_set():
                        with state.condition:
                            state.condition.wait_for(lambda: state.sequence != last or state.stop.is_set(), timeout=5)
                            frame, last = state.frame, state.sequence
                        if not frame or state.stop.is_set():
                            continue
                        self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n"+f"Content-Length: {len(frame)}\r\n\r\n".encode()+frame+b"\r\n")
                        self.wfile.flush()
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
    return Handler


def main():
    from pi_telemetry import Telemetry
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", required=True)
    parser.add_argument("--allow-client")
    parser.add_argument("--allow-subnet", help="auto (private LAN) or an explicit private CIDR")
    parser.add_argument("--hostnames", default="rover-pi.local")
    parser.add_argument("--access-key-file", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--camera", default="http://127.0.0.1:8080/stream.mjpg")
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--printed-classifier", type=Path)
    parser.add_argument("--default-mode", choices=("general", "printed"), default="general")
    parser.add_argument("--verifier", type=Path)
    parser.add_argument("--capture-dir", type=Path, default=Path("/home/espiob/rover/training-captures"))
    args = parser.parse_args()
    if not args.allow_client and not args.allow_subnet:
        parser.error("Specify --allow-client or --allow-subnet")
    if args.allow_subnet and not args.access_key_file:
        parser.error("LAN access requires --access-key-file")
    if args.default_mode == "printed" and not args.printed_classifier:
        parser.error("Printed mode requires --printed-classifier")
    access = None
    if args.access_key_file:
        from viewer_access import ViewerAccess
        access = ViewerAccess(args.access_key_file)
    detector = FastLeafDetector(str(args.detector), threads=2)
    config = json.loads((args.classifier.parent/"config.json").read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.classifier), sess_options=options, providers=["CPUExecutionProvider"])
    printed_session = printed_config = None
    if args.printed_classifier:
        printed_config = json.loads((args.printed_classifier.parent/"config.json").read_text(encoding="utf-8"))
        if (printed_config.get("scope") != "known_printed_demo"
                or printed_config.get("status") != "EXPERIMENTAL_KNOWN_PRINTS"
                or printed_config.get("classes") != config["classes"]
                or any(printed_config.get(key) != config[key]
                       for key in ("resize_short_edge", "center_crop", "mean", "std"))):
            parser.error("Printed classifier must be the matching experimental known-print demo")
        printed_options = ort.SessionOptions()
        printed_options.intra_op_num_threads = printed_options.inter_op_num_threads = 1
        printed_session = ort.InferenceSession(str(args.printed_classifier), sess_options=printed_options,
                                              providers=["CPUExecutionProvider"])
    verifier = None
    if args.verifier:
        from leaf_verifier import LeafVerifier
        verifier = LeafVerifier(args.verifier)
    state = State()
    state.available_modes = ("general", "printed") if printed_session is not None else ("general",)
    state.analysis_mode = args.default_mode
    try:
        from training_capture import TrainingCaptureStore
        args.capture_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=args.capture_dir):
            pass
        state.capture_store = TrainingCaptureStore(args.capture_dir)
    except (OSError, ValueError):
        state.capture_error = "No está disponible el guardado de ejemplos en la Raspberry."
    log = "/home/espiob/rover/logs/motion-consumption-"+time.strftime("%Y%m%d-%H%M%S")+".csv"
    state.telemetry = Telemetry(state, log)
    threads = [threading.Thread(target=fn, args=parameters, daemon=True) for fn, parameters in
               [(read_camera, (state, args.camera)), (detect_loop, (state, detector)),
                (classify_loop, (state, session, config, verifier, printed_session, printed_config)),
                (state.telemetry.run, ())]]

    server = ThreadingHTTPServer((args.bind, args.port), make_handler(state, args, access))
    server.daemon_threads = True
    for thread in threads:
        thread.start()

    def shutdown(*_):
        state.stop.set()
        with state.condition:
            state.condition.notify_all()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(f"Motion viewer http://{args.bind}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        state.stop.set()
        for thread in threads:
            thread.join(timeout=3)
        server.server_close()


if __name__ == "__main__":
    main()
