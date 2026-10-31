"""Describe the deployed pipeline on authorized phone photos of printed leaves.

No training, threshold selection, accuracy claim, Pi access or source edits.
Independent stills exercise the per-frame downstream policy, not the tracker.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont, ImageOps

from fast_leaf_detector import FastLeafDetector
from leaf_verifier import LeafVerifier
from rover_motion import capture_ready, classifier_input, classify_scores, leaf_crop, sharpness

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT/"reports/user_photo_review_20261026/manifest.json"
STATE = ROOT/"docs/overnight_state.json"
DESTINATION = ROOT/"reports/user_print_baseline_20261026"
MODELS = {"detector320": ROOT/"models/leaf-yolo/leaf_yolo_320.onnx",
          "verifier": ROOT/"models/leaf-verifier-candidate/model.onnx",
          "disease": ROOT/"models/potato-robust-candidate/model.onnx"}
CONFIG_HASHES = {"verifier": "a47afeb970a90872464ea16dd349033a9e4437015ff4b65d8dd127c3564e8276",
                 "disease": "30ef816f0f70eb3a318d42d13aa97a6dfb2bfb96d2591a657f790ea84e5d67b2"}
SCREEN_NAMES = {f"IMG_{index}.HEIC" for index in range(5171, 5177)}
STAGE_LABELS = {"no_detection": "Sin propuesta", "capture_not_ready": "Captura no apta",
                "verifier_rejected": "Verificador rechaza", "uncertain_disease": "Clase no concluyente",
                "diagnosed": "Clase orientativa"}
COLORS = {"capture_not_ready": "#e69c13", "verifier_rejected": "#cf4b45",
          "uncertain_disease": "#9570db", "diagnosed": "#22ad8a"}


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def category(row):
    if row["name"] in SCREEN_NAMES:
        return "excluded_screen", "Foto de pantalla/visor; no es entrada directa de cámara de hoja"
    if row["name"].lower().startswith("15be2e05"):
        return "excluded_setup", "Foto general del montaje; sin hoja impresa evaluable en primer plano"
    return "direct_print", "Foto directa de recortes impresos; etiqueta de enfermedad sin confirmar"


def font(size):
    for path in (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf")):
        if path.exists(): return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def annotate(image, row, destination):
    canvas = Image.new("RGB", (max(480, image.width), image.height+70), "#15252e")
    canvas.paste(image, (0, 70))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), row["name"][:60], fill="white", font=font(15))
    draw.text((8, 28), f"{len(row['detections'])} propuestas | Etiqueta real desconocida", fill="#dce8e5", font=font(14))
    if not row["detections"]: draw.text((8, 48), "Sin detección con umbral .45", fill="#f4bd59", font=font(14))
    for item in row["detections"]:
        x1,y1,x2,y2 = item["box"]
        color = COLORS[item["stage"]]
        draw.rectangle((x1,y1+70,x2,y2+70), outline=color, width=3)
        text = f"#{item['index']} {STAGE_LABELS[item['stage']]}"
        if item.get("prediction", {}).get("class"):
            text = f"#{item['index']} {item['prediction']['label']}"
        width = min(canvas.width-8, draw.textbbox((0,0),text,font=font(14))[2]+8)
        left = max(0,min(x1,canvas.width-width))
        top = max(70,y1+70-21)
        draw.rectangle((left,top,left+width,top+21), fill="#14242d")
        draw.text((left+3,top+1),text,fill=color,font=font(14))
    path = destination/"outputs/annotated"/f"{row['id']}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=91)
    return str(path.relative_to(destination)).replace("\\", "/")


def write_gallery(destination, rows, summary):
    cards = []
    for row in rows:
        if row["category"] != "direct_print": continue
        lines = []
        for item in row.get("detections", []):
            gate = item.get("verification")
            gate_text = f"{gate['leaf_probability']:.3f} (umbral .95)" if gate else "no ejecutado por calidad"
            prediction = item.get("prediction")
            diagnosis = prediction["label"] if prediction else "Clasificador no ejecutado"
            q = item["quality"]
            lines.append(f"<li><a href='{html.escape(item['crop_file'])}'>Recorte #{item['index']}</a>: "
                         f"detector {item['score']:.3f}; nitidez {q['laplacian_variance']:.1f}; "
                         f"lado mínimo {q['minimum_side_pixels']:.1f}px; borde {'completo' if q['border_complete'] else 'cortado'}; "
                         f"gate {gate_text}. {html.escape(diagnosis)}</li>")
        path = html.escape(row.get("annotated", ""))
        cards.append(f"<article><h2>{html.escape(row['name'])}</h2><a href='{path}'><img src='{path}' loading='lazy' alt='Propuestas y etapas sobre la foto'></a>"
                     f"<p>{len(row.get('detections', []))} propuestas. Estado: {html.escape(', '.join(STAGE_LABELS[k] for k in row.get('stages', ['no_detection'])))}</p>"
                     f"<ul>{''.join(lines)}</ul></article>")
    excluded = "".join(f"<li>{html.escape(row['name'])}: {html.escape(row['category_reason'])}</li>" for row in rows if row["category"] != "direct_print")
    text = f"""<!doctype html><html lang='es'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Línea base de los impresos</title><style>body{{font:16px system-ui;background:#eef2f0;color:#193b32;margin:24px}}h1{{font-size:30px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:20px}}article{{background:white;padding:16px;border-radius:12px}}h2{{font-size:16px;overflow-wrap:anywhere}}img{{width:100%;max-height:640px;object-fit:contain}}li{{margin:8px 0}}a{{color:#17684e}}p{{line-height:1.5}}header{{max-width:1000px;margin-bottom:22px}}</style>
<header><h1>Línea base: fotos de los impresos</h1><p>{summary['included_photos']} fotos directas; {summary['excluded_photos']} exclusiones. Modelos actuales, sin entrenar ni modificar umbrales.</p>
<p>Las clases son predicciones orientativas. Falta confirmar las etiquetas de los diseños; no se calcula precisión ni se cuenta como resultado correcto una abstención.</p>
<p>Las fotos conservan proporción y orientación, con lado máximo640px. Son fotos del teléfono, no video de la Pi; no se evalúan seguimiento, movimiento ni latencia real del carro.</p>
<p><a href='summary.json'>Resumen y hashes</a> · <a href='results.json'>Detalle por foto</a> · <a href='evaluation_manifest.json'>Manifiesto y exclusiones</a></p><details><summary>Archivos excluidos</summary><ul>{excluded}</ul></details></header><main class='grid'>{''.join(cards)}</main></html>"""
    (destination/"index.html").write_text(text, encoding="utf-8")


def evaluate(manifest_path=MANIFEST, destination=DESTINATION):
    started = time.monotonic()
    if destination.exists(): raise FileExistsError("Preserve the existing baseline; use a new output directory")
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    deployed = json.loads(STATE.read_text(encoding="utf-8"))
    if len(source["images"]) != 52: raise ValueError("Expected the authorized52-image review manifest")
    protected = {str(path): digest(path) for path in (manifest_path, STATE, *MODELS.values(), *(MODELS[key].with_name("config.json") for key in CONFIG_HASHES),
                 ROOT/"rover_motion.py", ROOT/"fast_leaf_detector.py", ROOT/"leaf_verifier.py", Path(__file__))}
    for name,path in MODELS.items():
        if digest(path) != deployed["model_hashes"][name]: raise ValueError("Local model does not match deployed SHA")
    for name,checksum in CONFIG_HASHES.items():
        if digest(MODELS[name].with_name("config.json")) != checksum: raise ValueError("Production config changed")
    if digest(ROOT/"rover_motion.py") != deployed["active_runtime_sha256"]: raise ValueError("Runtime source differs from deployed version")
    detector = FastLeafDetector(MODELS["detector320"], threads=2)
    verifier = LeafVerifier(MODELS["verifier"], threads=1)
    if verifier.threshold != .95: raise ValueError("Production gate must remain .95")
    config = json.loads(MODELS["disease"].with_name("config.json").read_text())
    options=ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
    session=ort.InferenceSession(str(MODELS["disease"]), sess_options=options, providers=["CPUExecutionProvider"])
    input_name=session.get_inputs()[0].name
    destination.mkdir(parents=True)
    (destination/"outputs/crops").mkdir(parents=True)
    rows=[]
    for index,entry in enumerate(source["images"], 1):
        selected, reason = category(entry)
        path=(ROOT/entry["preview"]).resolve(strict=True)
        if not path.is_relative_to(manifest_path.parent.resolve()/"previews"): raise ValueError("Unexpected preview location")
        row={"id": f"{index:03d}", "name": entry["name"], "preview": entry["preview"], "preview_sha256": digest(path),
             "original_source_sha256": entry["source_sha256"], "category": selected, "category_reason": reason,
             "disease_label": entry.get("disease_label"), "ground_truth_confirmed": False, "detections": []}
        rows.append(row)
        if selected != "direct_print": continue
        if time.monotonic()-started > 110: raise TimeoutError("Two-minute processing budget approached; preserve partial evidence")
        with Image.open(path) as original:
            image=ImageOps.exif_transpose(original).convert("RGB")
        row["oriented_preview_size"] = list(image.size)
        image.thumbnail((640,640), Image.Resampling.BILINEAR)
        row["input_frame_size"] = list(image.size)
        tick=time.perf_counter(); detections=detector.detect(image)
        row["pc_detector_ms"]=(time.perf_counter()-tick)*1000
        for number,detected in enumerate(detections, 1):
            item={**detected,"index":number,"stage":"capture_not_ready","verification":None}
            crop=leaf_crop(image, detected["box"])
            x1,y1,x2,y2=detected["box"]
            quality=bool(capture_ready(image, detected["box"], crop))
            item["quality"]={"accepted":quality,"laplacian_variance":sharpness(crop),
                             "minimum_side_pixels":min(x2-x1,y2-y1),
                             "border_complete":bool(x1>=3 and y1>=3 and x2<=image.width-3 and y2<=image.height-3)}
            item["crop_file"]=f"outputs/crops/{row['id']}_{number}.jpg"
            crop.save(destination/item["crop_file"],quality=93)
            if quality:
                item["verification"]=verifier.verify(crop)
                item["stage"]="verifier_rejected"
                if item["verification"]["accepted"]:
                    logits=session.run(None,{input_name:classifier_input(crop,config)})[0]
                    if logits.shape != (1,3) or not np.isfinite(logits).all(): raise ValueError("Invalid disease logits")
                    item["prediction"]=classify_scores(logits,config["classes"])
                    item["stage"]="diagnosed" if item["prediction"]["class"] is not None else "uncertain_disease"
            row["detections"].append(item)
        row["stages"]=sorted({item["stage"] for item in row["detections"]}) or ["no_detection"]
        row["annotated"]=annotate(image,row,destination)
        if index%10==0: print(json.dumps({"photos_visited":index,"total":52}),flush=True)
    included=[row for row in rows if row["category"]=="direct_print"]
    detections=[item for row in included for item in row["detections"]]
    photo_stages=Counter(stage for row in included for stage in row["stages"])
    stages=Counter(item["stage"] for item in detections)
    predictions=Counter(item["prediction"]["class"] for item in detections if item["stage"]=="diagnosed")
    summary={"created_utc":datetime.now(timezone.utc).isoformat(),"completed":True,"processing_seconds":round(time.monotonic()-started,3),
             "input_photos":len(rows),"included_photos":len(included),"excluded_photos":len(rows)-len(included),
             "photos_with_proposals":sum(bool(row["detections"]) for row in included),"photos_without_proposals":sum(not row["detections"] for row in included),
             "proposals":len(detections),"proposal_stages":dict(stages),"photo_stages_nonexclusive":dict(photo_stages),
             "predicted_classes_confident":dict(predictions),"known_disease_labels":0,"accuracy":None,
             "thresholds":{"detector_confidence":detector.threshold,"nms_iou":detector.iou_threshold,"gate":verifier.threshold,
                           "disease_softmax":.7,"disease_margin":.2,"quality_min_side_px":40,"quality_min_sharpness":30,"quality_border_margin_px":3},
             "model_hashes":{name:digest(path) for name,path in MODELS.items()},"protected_inputs":protected,
             "geometry":"Pre-oriented preview; preserve aspect ratio and fit within640x640 without upscaling, cropping or stretching; detector runtime letterbox320",
             "pipeline":"FastLeafDetector→leaf_crop12%/gray-square→capture_ready→LeafVerifier.95→classifier_input/classify_scores",
             "tracker_evaluated":False,"training":False,"threshold_tuning":False,"pi_accessed":False,"originals_modified":False,
             "limitations":["Unknown design labels; no disease accuracy, precision or recall is estimable",
                 "Repeated views of few printed designs are correlated and are not52 independent biological leaves",
                 "Portrait phone images and frozen JPEG previews differ from Pi camera640x360 and lighting",
                 "Per-frame downstream stages only; no track confirmation, temporal expiry or moving-rover latency",
                 "Non-leaf proposals cannot be labeled false positives without manual bounding-box ground truth",
                 "Softmax and leaf scores are uncalibrated outputs, not disease probability"]}
    if any(digest(Path(path))!=checksum for path,checksum in protected.items()): raise RuntimeError("Protected input changed during baseline")
    save(destination/"evaluation_manifest.json",{"source_manifest_sha256":digest(manifest_path),"rows":[{key:value for key,value in row.items() if key not in ("detections",)} for row in rows]})
    save(destination/"results.json",{"photos":rows})
    save(destination/"summary.json",summary)
    write_gallery(destination,rows,summary)
    print(json.dumps({key:summary[key] for key in ("completed","processing_seconds","included_photos","excluded_photos","proposals","proposal_stages","predicted_classes_confident")}),flush=True)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=MANIFEST)
    parser.add_argument("--output",type=Path,default=DESTINATION)
    args=parser.parse_args()
    evaluate(args.manifest.resolve(),args.output.resolve())
