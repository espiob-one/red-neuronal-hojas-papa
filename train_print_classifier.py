"""One fixed four-epoch print-adaptation experiment, isolated from production.

All supplied print crops are development training material. Only reused public
validation diagnostics select a checkpoint; no new independent test is claimed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler

import train_robust_classifier as base
from train_motion_classifier import Validation, measure
from evaluate_robustness_suite import CONDITIONS, classification_metrics
from rover_motion import classifier_input

ROOT=Path(__file__).resolve().parent
SEED=2026102601
USER_MANIFEST=ROOT/"data/user_prints_20261026/training_manifest.json"
USER_MANIFEST_SHA="cb9b70b54b2194d05f62fdff4c4295b8a1ce9332ec429dcf761bb82ef4e35cee"
SOURCE=ROOT/"runs/robust_20261021_232142/best_weights.pt"
SOURCE_SHA="2b6b1b4985480472cce6bc7f72d74112bc11179ea5cd29eb97caff1748401029"
REPORTS=ROOT/"reports/print_adaptation_20261026"
OUTPUT=ROOT/"models/potato-print-candidate-20261026"
RUNTIME_CONFIG={"resize_short_edge":232,"center_crop":224,"mean":base.MEAN,"std":base.STD}


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def guards(candidate,reference,classes):
    clean=candidate["clean"]["accuracy"]>=reference["clean"]["accuracy"]-.005-1e-12
    recall=all(candidate[c]["per_class"][label]["recall"]>=reference[c]["per_class"][label]["recall"]-.03-1e-12
               for c in CONDITIONS for label in classes)
    macro=all(candidate[c]["macro_f1"]>=reference[c]["macro_f1"]-.01-1e-12 for c in CONDITIONS)
    errors=sum(candidate[c]["high_confidence_errors"] for c in CONDITIONS)
    reference_errors=sum(reference[c]["high_confidence_errors"] for c in CONDITIONS)
    return {"clean_accuracy_loss_le_0.005":bool(clean),"each_class_condition_recall_loss_le_0.03":bool(recall),
            "each_condition_macro_f1_loss_le_0.01":bool(macro),"confident_errors_total_not_increased":errors<=reference_errors,
            "confident_errors_total":errors,"reference_confident_errors_total":reference_errors,
            "confident_errors_by_condition":{c:candidate[c]["high_confidence_errors"] for c in CONDITIONS},
            "reference_confident_errors_by_condition":{c:reference[c]["high_confidence_errors"] for c in CONDITIONS},
            "mean_public_macro_f1":float(np.mean([candidate[c]["macro_f1"] for c in CONDITIONS])),
            "reference_mean_public_macro_f1":float(np.mean([reference[c]["macro_f1"] for c in CONDITIONS])),
            "eligible":bool(clean and recall and macro and errors<=reference_errors)}


def freeze(model):
    for parameter in model.parameters(): parameter.requires_grad=False
    for parameter in model.features[-3:].parameters(): parameter.requires_grad=True
    for parameter in model.classifier.parameters(): parameter.requires_grad=True
    for layer in model.modules():
        if isinstance(layer,nn.BatchNorm2d):
            layer.eval()
            for parameter in layer.parameters(): parameter.requires_grad=False
    return [{"params":[p for p in model.features[-3:].parameters() if p.requires_grad],"lr":5e-5},
            {"params":[p for p in model.classifier.parameters() if p.requires_grad],"lr":1e-4}]


def bn_snapshot(model):
    return {name:value.detach().cpu().clone() for name,value in model.named_buffers()
            if name.endswith(("running_mean","running_var","num_batches_tracked"))}


def buffer_fingerprints(snapshot):
    return {name:{"shape":list(value.shape),"dtype":str(value.dtype),
                  "sha256":hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()} for name,value in snapshot.items()}


class MixedBatches(Sampler):
    """Exactly24 public and8 print samples per32, balanced labels, no weights."""
    def __init__(self,public_rows,print_rows):
        self.public={label:[i for i,row in enumerate(public_rows) if row["class_index"]==label] for label in range(3)}
        self.prints={label:[len(public_rows)+i for i,row in enumerate(print_rows) if row["class_index"]==label] for label in range(3)}
        if not all(self.public.values()) or not all(self.prints.values()): raise ValueError("Every source must contain all3 classes")
        self.batches=math.ceil(len(public_rows)/24)
        self.epoch=0
    def __len__(self): return self.batches
    def __iter__(self):
        rng=random.Random(SEED+self.epoch)
        for step in range(self.batches):
            indices=[rng.choice(self.public[label]) for label in range(3) for _ in range(8)]
            indices += [rng.choice(self.prints[(step*8+k)%3]) for k in range(8)]
            rng.shuffle(indices)
            yield indices


class MixedTraining(Dataset):
    def __init__(self,public,print_rows):
        self.public,self.print_rows=public,print_rows
        self.prints=[]
        for row in print_rows:
            with Image.open(ROOT/row["path"]) as image: image=image.convert("RGB")
            if image.width!=image.height: raise ValueError("Print ROI must already use runtime square padding")
            self.prints.append(image)
    def __len__(self): return len(self.public)+len(self.print_rows)
    def __getitem__(self,index):
        if index<len(self.public): return self.public[index]
        index-=len(self.public)
        image=self.prints[index].copy()
        if random.random()<.5: image=ImageOps.mirror(image)
        image=image.rotate(random.uniform(-8,8),resample=Image.Resampling.BILINEAR,fillcolor=(128,128,128))
        image=ImageEnhance.Brightness(image).enhance(random.uniform(.92,1.08))
        image=ImageEnhance.Contrast(image).enhance(random.uniform(.92,1.08))
        image=ImageEnhance.Color(image).enhance(random.uniform(.95,1.05))
        tensor=torch.from_numpy(classifier_input(image,RUNTIME_CONFIG)[0])
        return tensor,self.print_rows[index]["class_index"]


def self_test():
    classes=["early","late","healthy"]
    reference={c:{"accuracy":.97,"macro_f1":.95,"high_confidence_errors":5,
                   "per_class":{label:{"recall":.94} for label in classes}} for c in CONDITIONS}
    assert guards(reference,reference,classes)["eligible"]
    for key in ("clean","motion_blur","gamma_rgb"):
        candidate=copy.deepcopy(reference); candidate[key]["per_class"][classes[0]]["recall"]-=.03001
        assert not guards(candidate,reference,classes)["eligible"]
    for condition in CONDITIONS:
        candidate=copy.deepcopy(reference); candidate[condition]["macro_f1"]-=.01001
        assert not guards(candidate,reference,classes)["eligible"]
    candidate=copy.deepcopy(reference); candidate["clean"]["accuracy"]-=.005
    assert guards(candidate,reference,classes)["eligible"]
    candidate["clean"]["accuracy"]-=.00001
    assert not guards(candidate,reference,classes)["eligible"]
    candidate=copy.deepcopy(reference); candidate["jpeg_occlusion"]["high_confidence_errors"]+=1
    assert not guards(candidate,reference,classes)["eligible"]
    public=[{"class_index":i%3} for i in range(72)]
    prints=[{"class_index":i} for i in range(3)]
    sampler=MixedBatches(public,prints)
    for batch in sampler:
        assert len(batch)==32 and sum(i>=72 for i in batch)==8
        assert Counter(public[i]["class_index"] for i in batch if i<72)=={0:8,1:8,2:8}
    first=list(sampler); sampler.epoch=1
    assert first!=list(sampler)
    model=base.models.mobilenet_v2(weights=None); freeze(model)
    assert all(not p.requires_grad for layer in model.modules() if isinstance(layer,nn.BatchNorm2d) for p in layer.parameters())
    image=Image.fromarray(np.random.default_rng(9).integers(0,256,(256,256,3),dtype=np.uint8))
    expected=classifier_input(image,RUNTIME_CONFIG)[0]
    actual=base.transforms.Compose([base.transforms.Resize(232),base.transforms.CenterCrop(224),base.transforms.ToTensor(),base.transforms.Normalize(base.MEAN,base.STD)])(image).numpy()
    np.testing.assert_allclose(expected,actual,rtol=0,atol=5e-7)
    return {"passed":True,"guard_boundaries":True,"all_condition_guards":True,"confident_error_guard":True,
            "exact25pct_print_sampling":True,"class_balanced_replay":True,"bn_affine_frozen":True,
            "runtime_preprocess_parity":True,"training_executed":False}


def load_inputs():
    if digest(USER_MANIFEST)!=USER_MANIFEST_SHA or digest(SOURCE)!=SOURCE_SHA: raise ValueError("Fixed manifest or source changed")
    users=json.loads(USER_MANIFEST.read_text(encoding="utf-8"))
    classes=json.loads((ROOT/"data/metadata/classes.json").read_text())
    if users["classes"]!=classes or len(users["images"])!=37: raise ValueError("Unexpected reviewed print dataset")
    print_rows=users["images"]
    for row in print_rows:
        if (row.get("split")!="train_development" or row.get("training_authorized") is not True
                or row.get("independent_evaluation_eligible") is not False or row.get("visual_roi_reviewed") is not True
                or row.get("label_source")!="user_confirmation_2026-10-26" or classes[row["class_index"]]!=row["label"]):
            raise ValueError("Print class, authorization or development grouping changed")
        path=(ROOT/row["path"]).resolve(strict=True)
        if not path.is_relative_to(USER_MANIFEST.parent/"crops") or digest(path)!=row["sha256"]: raise ValueError("Reviewed ROI changed")
    for group in {r["group_id"] for r in print_rows}:
        if len({r["label"] for r in print_rows if r["group_id"]==group})!=1: raise ValueError("Mixed labels in design family")
    public_path=ROOT/"data/metadata/manifest.json"
    public=json.loads(public_path.read_text())
    parts={split:[row for row in public if row["split"]==split] for split in ("train","val","test")}
    for field in ("group_id","sha256","pixel_sha256"):
        sets={split:{r[field] for r in rows} for split,rows in parts.items()}
        if any(sets[a]&sets[b] for a,b in (("train","val"),("train","test"),("val","test"))): raise ValueError("Public partitions overlap")
    audit=[]
    for row in parts["train"]+parts["val"]:
        if not row["path"].startswith("data/raw/") or "healthy_error" in row["path"]: raise ValueError("Unexpected public input")
        if digest(ROOT/row["path"])!=row["sha256"]: raise ValueError("Public RGB changed")
        mask=ROOT/"data/segmented"/row["label"]/(Path(row["path"]).stem+"_final_masked.jpg")
        audit.append({"path":row["path"],"rgb_sha256":row["sha256"],"split":row["split"],"group_id":row["group_id"],
                      "mask_path":str(mask.relative_to(ROOT)),"mask_sha256":digest(mask)})
    return classes,parts,print_rows,audit


def evaluate_user_fit(model,print_rows,classes,device):
    """Called only after checkpoint selection; training-fit description only."""
    model.eval(); scores=[]
    with torch.inference_mode():
        for row in print_rows:
            with Image.open(ROOT/row["path"]) as image: tensor=classifier_input(image.convert("RGB"),RUNTIME_CONFIG)
            scores.append(model(torch.from_numpy(tensor).to(device)).softmax(1).cpu().numpy()[0])
    result=classification_metrics(print_rows,np.asarray(scores),classes)
    result.update(scope="Training/development fit only; repeated designs, no independent validation",
                  independent_accuracy_claim=False,used_for_checkpoint_selection=False)
    records=[]
    for row,probabilities in zip(print_rows,scores):
        records.append({"id":row["id"],"path":row["path"],"group_id":row["group_id"],"declared_class":row["label"],
                        "predicted_class":classes[int(probabilities.argmax())],"scores":[float(value) for value in probabilities]})
    return {"metrics":result,"rows":records}


def verify_export_decisions(path,loaders,selected_predictions):
    """All reused public inputs: export must preserve argmax and .7/.2 policy."""
    import onnxruntime as ort
    options=ort.SessionOptions(); options.intra_op_num_threads=1; options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"])
    per_condition={}; maximum=0.
    def decision(scores):
        ordered=np.sort(scores)
        return int(np.argmax(scores)),bool(ordered[-1]>=.7 and ordered[-1]-ordered[-2]>=.2)
    for condition,loader in loaders.items():
        count,argmax_errors,policy_errors=0,0,0
        for index,expected in enumerate(selected_predictions[condition]):
            tensor=loader.dataset[index][0].unsqueeze(0).numpy()
            logits=session.run(None,{session.get_inputs()[0].name:tensor})[0][0]
            values=np.exp(logits-logits.max()); scores=values/values.sum()
            maximum=max(maximum,float(np.abs(scores-np.asarray(expected)).max()))
            a,b=decision(scores),decision(np.asarray(expected))
            argmax_errors+=a[0]!=b[0]; policy_errors+=a!=b; count+=1
        per_condition[condition]={"inputs":count,"argmax_mismatches":argmax_errors,"decision_mismatches":policy_errors}
    return {"passed":maximum<=1e-4 and all(row["argmax_mismatches"]==row["decision_mismatches"]==0 for row in per_condition.values()),
            "inputs":sum(row["inputs"] for row in per_condition.values()),"max_probability_difference":maximum,
            "conditions":per_condition,"policy":{"softmax":.7,"margin":.2},"test_pixels_used":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--train",action="store_true",help="Run the one authorized fixed4-epoch candidate")
    args=parser.parse_args(argv)
    check=self_test()
    if args.self_test:
        print(json.dumps(check)); return 0
    if not args.train: parser.error("Use --self-test or explicitly --train; no other training configuration is exposed")
    if REPORTS.exists() or OUTPUT.exists() or any((ROOT/"runs").glob("print_adaptation_*")):
        raise FileExistsError("A print adaptation experiment already exists; this protocol allows one configuration only")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; do not silently change execution")
    torch.set_num_threads(3)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    device=torch.device("cuda")
    classes,parts,print_rows,audit=load_inputs()
    run=ROOT/"runs"/("print_adaptation_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    run.mkdir(parents=True,exist_ok=False); REPORTS.mkdir(parents=True,exist_ok=False)
    protected=[USER_MANIFEST,SOURCE,ROOT/"data/metadata/manifest.json",ROOT/"data/metadata/classes.json",
               ROOT/"docs/planes/ADAPTACION_IMPRESOS_20261026.md",Path(__file__),Path(base.__file__),
               ROOT/"train_motion_classifier.py",ROOT/"evaluate_robustness_suite.py",ROOT/"rover_motion.py"]
    hashes={str(path):digest(path) for path in protected}
    config={"architecture":"mobilenet_v2","classes":classes,"epochs":4,"batch_size":32,"seed":SEED,
            **RUNTIME_CONFIG,"square_pad":True,"square_pad_color":[128,128,128],"crop_margin":.12,
            "source_weights":str(SOURCE),"source_weights_sha256":SOURCE_SHA,"print_manifest_sha256":USER_MANIFEST_SHA,
            "train_last_feature_blocks":3,"feature_lr":5e-5,"classifier_lr":1e-4,"weight_decay":1e-4,
            "batchnorm_running_and_affine_frozen":True,"sampling":"Exact24 public +8 print per32; balanced class sampling only; unweighted CE",
            "print_augmentation":"mirror.5,rotation±8deg,brightness/contrast.92-1.08,saturation.95-1.05; same for all labels",
            "public_replay":"Existing40%original/60%composed robust training augmentation unchanged",
            "conditions":list(CONDITIONS),"selection":"Eligible epoch with highest mean public-validation macroF1; user fit only after selection",
            "guards":{"clean_accuracy_max_loss":.005,"per_class_condition_recall_max_loss":.03,"per_condition_macro_f1_max_loss":.01,
                      "high_confidence_errors_total_max_increase":0},
            "split_counts_public":{key:len(value) for key,value in parts.items()},"user_crops":len(print_rows),
            "user_classes":dict(Counter(row["label"] for row in print_rows)),"user_design_groups":sorted({r["group_id"] for r in print_rows}),
            "test_pixels_used":False,"user_data_role":"ALL training/development; no random split or independent test",
            "printed_source_ids_unknown":True,"complete_leakage_absence_claimed":False,"deployed":False,"hashes":hashes}
    save(run/"config.json",config); save(REPORTS/"protocol.json",config); save(REPORTS/"self_test.json",check)
    save(REPORTS/"public_inputs.json",{"files":audit,"test_pixels_read":False,"test_metadata_only_for_overlap_audit":True})
    print(json.dumps({"stage":"preparing","run":str(run),"epochs":4,"classes":config["user_classes"]}),flush=True)
    train_public=base.LeafDataset(parts["train"],"train")
    mixed=MixedTraining(train_public,print_rows)
    sampler=MixedBatches(parts["train"],print_rows)
    train_loader=DataLoader(mixed,batch_sampler=sampler,num_workers=0,pin_memory=True)
    assets=base.LeafDataset(parts["val"],"train")
    valsets={condition:Validation(parts["val"],condition,list(zip(assets.originals,assets.leaves))) for condition in CONDITIONS}
    loaders={condition:DataLoader(dataset,batch_size=64,num_workers=0,pin_memory=True) for condition,dataset in valsets.items()}
    model=base.create_model(classes,SOURCE).to(device)
    reference,reference_predictions=measure(model,loaders,device,classes)
    save(run/"reference_public.json",reference); save(run/"reference_public_predictions.json",reference_predictions)
    optimizer=torch.optim.AdamW(freeze(model),weight_decay=1e-4)
    frozen_bn=bn_snapshot(model)
    frozen_bn_hashes=buffer_fingerprints(frozen_bn)
    save(run/"batchnorm_frozen_before.json",frozen_bn_hashes)
    history=[]; best=None; training_started=time.perf_counter()
    for epoch in range(1,5):
        sampler.epoch=epoch
        model.train()
        for layer in model.modules():
            if isinstance(layer,nn.BatchNorm2d): layer.eval()
        total_loss=0.; batches=0
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(images),labels)
            loss.backward(); optimizer.step()
            total_loss+=float(loss.detach()); batches+=1
        current_bn_hashes=buffer_fingerprints(bn_snapshot(model))
        if frozen_bn_hashes!=current_bn_hashes:
            raise RuntimeError("Frozen BN state changed")
        save(run/f"epoch_{epoch}_batchnorm.json",{"buffers_byte_identical":True,"hashes":current_bn_hashes})
        metrics,predictions=measure(model,loaders,device,classes)
        decision=guards(metrics,reference,classes)
        record={"epoch":epoch,"mean_train_loss":total_loss/batches,"batches":batches,
                "public_samples":batches*24,"print_samples":batches*8,"metrics":metrics,"guards":decision,
                "elapsed_training_seconds":time.perf_counter()-training_started}
        history.append(record)
        torch.save(model.state_dict(),run/f"epoch_{epoch}_weights.pt")
        save(run/f"epoch_{epoch}_public_predictions.json",predictions); save(run/"history.json",history)
        if decision["eligible"] and (best is None or decision["mean_public_macro_f1"]>best["guards"]["mean_public_macro_f1"]):
            best=record
            torch.save(model.state_dict(),run/"best_weights.pt")
        print(json.dumps({"epoch":epoch,"loss":record["mean_train_loss"],"guards":decision,
                          "elapsed_training_seconds":record["elapsed_training_seconds"]}),flush=True)
    selected=best if best is not None else history[-1]
    checkpoint=run/("best_weights.pt" if best is not None else "epoch_4_weights.pt")
    model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location=device))
    save(run/"selection_frozen_before_user_fit.json",{"selected_epoch":selected["epoch"],"checkpoint_sha256":digest(checkpoint),
             "public_guards_passed":best is not None,"reason":"highest eligible mean public F1" if best else "No epoch eligible; last epoch retained as REJECTED diagnostic",
             "user_metrics_computed_yet":False})
    # User-development scoring in this experiment begins only after selection.
    selected_fit=evaluate_user_fit(model,print_rows,classes,device)
    baseline=base.create_model(classes,SOURCE).to(device)
    baseline_fit=evaluate_user_fit(baseline,print_rows,classes,device)
    save(REPORTS/"user_development_fit.json",{"role":"Training fit ONLY; not independent accuracy; no epoch selection",
              "baseline":baseline_fit,"selected_candidate":selected_fit})
    config.update(selected_epoch=selected["epoch"],public_validation_guards_passed=best is not None,
                  status="ELIGIBLE_OFFLINE_ONLY" if best else "REJECTED",checkpoint_sha256=digest(checkpoint))
    exported=None
    if best is not None:
        if OUTPUT.exists(): raise FileExistsError("Never overwrite candidate output")
        exported=base.export_onnx(model,OUTPUT,config,{"clean":loaders["clean"],"challenge":loaders["motion_blur"]})
        exported["onnx_sha256"]=digest(OUTPUT/"model.onnx")
        selected_predictions=json.loads((run/f"epoch_{selected['epoch']}_public_predictions.json").read_text())
        exported["all_public_decision_parity"]=verify_export_decisions(OUTPUT/"model.onnx",loaders,selected_predictions)
        save(OUTPUT/"all_public_decision_parity.json",exported["all_public_decision_parity"])
        if not exported["all_public_decision_parity"]["passed"]:
            config["status"]="EXPORT_PARITY_FAILED_OFFLINE_ONLY"
    if any(digest(Path(path))!=checksum for path,checksum in hashes.items()): raise RuntimeError("Frozen training source or manifest changed")
    if any(digest(ROOT/row["path"])!=row["sha256"] for row in print_rows): raise RuntimeError("User ROI changed")
    report={"created_utc":datetime.now(timezone.utc).isoformat(),"run":str(run),"status":config["status"],"public_guards_passed":best is not None,
            "reference_public":reference,"history":history,"selected_epoch":selected["epoch"],"selected_checkpoint":str(checkpoint),
            "selected_checkpoint_sha256":digest(checkpoint),"selection_metric":"mean macroF1 on6 reused public validation conditions only",
            "user_fit_used_for_selection":False,"user_fit_baseline":baseline_fit["metrics"],"user_fit_selected":selected_fit["metrics"],
            "export":exported,"output":str(OUTPUT) if exported else None,"config":config,
            "training_seconds":round(time.perf_counter()-training_started,3),"deployed":False,"pi_accessed":False,"test_pixels_used":False,
            "limitations":["All37 print crops were used for development training; final fit is not independent validation",
                 "Repeated views of3 design families do not show generalization to new prints or biological leaves",
                 "Printed source identity is unknown and may overlap PlantVillage; no complete source-leakage exclusion claim",
                 "Public validation conditions are reused diagnostic guards, not a newly untouched holdout",
                 "Detector/verifier/tracker unchanged; classifier adaptation does not solve border or motion quality failures",
                 "A public-guard PASS does not authorize deployment or prove field accuracy"]}
    save(run/"report.json",report); save(REPORTS/"report.json",report)
    save(run/"final_config.json",config)
    print(json.dumps({"run":str(run),"status":config["status"],"selected_epoch":selected["epoch"],
                      "user_training_fit_baseline":baseline_fit["metrics"]["accuracy"],"user_training_fit_candidate":selected_fit["metrics"]["accuracy"],
                      "exported":exported is not None,"deployed":False}),flush=True)
    return 0


if __name__=="__main__": raise SystemExit(main())
