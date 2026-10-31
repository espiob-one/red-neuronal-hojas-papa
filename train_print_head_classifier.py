"""One isolated8-epoch linear-head adaptation with frozen features and teacher.

Same reused public guards as the previous rejected experiment. User37 crops
are training/development only and are scored descriptively after selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset

import train_robust_classifier as base
from train_motion_classifier import Validation,measure
from train_print_classifier import (MixedTraining,MixedBatches,load_inputs,evaluate_user_fit,
                                    guards,verify_export_decisions,SOURCE,SOURCE_SHA,USER_MANIFEST,USER_MANIFEST_SHA)
from evaluate_robustness_suite import CONDITIONS

ROOT=Path(__file__).resolve().parent
REPORTS=ROOT/"reports/print_head_20261026"
OUTPUT=ROOT/"models/potato-print-head-candidate-20261026"
PROTOCOL=ROOT/"docs/planes/CABEZA_IMPRESOS_20261026.md"
SEED=2026102601  # Also the seed of the immutable replay batch sampler helper.
TEMPERATURE=2.
DISTILLATION_WEIGHT=10.


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def fingerprint(state):
    return {name:{"dtype":str(value.dtype),"shape":list(value.shape),
                  "sha256":hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            for name,value in state.items()}


def frozen_student(model):
    for parameter in model.parameters():parameter.requires_grad=False
    for parameter in model.classifier[1].parameters():parameter.requires_grad=True
    model.eval()  # Includes feature BN and Dropout; trained head is strictly linear.
    model.classifier[1].train()
    return model.classifier[1].parameters()


class ReplayDomains(Dataset):
    """Keep source membership through sampler shuffle; public KD never sees prints."""
    def __init__(self,mixed):self.mixed=mixed
    def __len__(self):return len(self.mixed)
    def __getitem__(self,index):
        image,label=self.mixed[index]
        return image,label,index<len(self.mixed.public)


def objective(student_logits,teacher_public_logits,labels,is_public):
    if int(is_public.sum())!=24 or len(labels)!=32 or teacher_public_logits.shape!=(24,3):
        raise ValueError("Each step requires exactly24 public/8 print, with independent teacher public logits")
    ce=F.cross_entropy(student_logits,labels)
    log_student=F.log_softmax(student_logits[is_public]/TEMPERATURE,dim=1)
    log_teacher=F.log_softmax(teacher_public_logits.detach()/TEMPERATURE,dim=1)
    # Literal root protocol: KL(student || teacher), not torch.kl_div's usual
    # target-teacher direction. No pseudo-labels or confidence filtering.
    kl=(log_student.exp()*(log_student-log_teacher)).sum(dim=1).mean()
    return ce+DISTILLATION_WEIGHT*TEMPERATURE**2*kl,ce,kl


def self_test():
    rng=torch.Generator().manual_seed(72)
    student=torch.randn(32,3,generator=rng,requires_grad=True)
    teacher=torch.randn(24,3,generator=rng,requires_grad=True)
    flags=torch.tensor([True]*24+[False]*8)
    labels=torch.arange(32)%3
    loss,ce,kl=objective(student,teacher,labels,flags)
    s=(student[flags]/2).softmax(1)
    reference=(s*(s.log()-(teacher.detach()/2).log_softmax(1))).sum(1).mean()
    torch.testing.assert_close(kl,reference)
    torch.testing.assert_close(loss,ce+40*kl)
    loss.backward()
    assert student.grad is not None and teacher.grad is None
    # A change to any print logit must not change the public distillation term.
    changed=student.detach().clone();changed[~flags]+=torch.tensor([2.,-1.,4.])
    torch.testing.assert_close(objective(changed,teacher.detach(),labels,flags)[2],kl.detach())
    model=base.models.mobilenet_v2(weights=None)
    frozen_student(model)
    assert {name for name,p in model.named_parameters() if p.requires_grad}=={"classifier.1.weight","classifier.1.bias"}
    assert not any(layer.training for layer in model.features.modules()) and not model.classifier[0].training
    assert not any(p.requires_grad for p in model.features.parameters())
    class IndexOnlyMixed:
        public=[None]*72
        def __len__(self):return 75
        def __getitem__(self,index):return torch.tensor(index),index%3
    index_domains=ReplayDomains(IndexOnlyMixed())
    shuffled=MixedBatches([{"class_index":i%3} for i in range(72)],
                          [{"class_index":i} for i in range(3)])
    saw_mixed_positions=False
    for indices,_,public in DataLoader(index_domains,batch_sampler=shuffled,num_workers=0):
        assert int(public.sum())==24 and torch.equal(public,indices<72)
        saw_mixed_positions |= not torch.equal(public,torch.tensor([True]*24+[False]*8))
    assert saw_mixed_positions
    return {"passed":True,"kl_direction":"student||detached_teacher","teacher_gradient_absent":True,
            "no_print_distillation":True,"only_linear_head_trainable":True,"all_features_bn_dropout_eval":True,
            "source_flags_survive_actual_batch_shuffle":True,
            "one_fixed8_epoch_configuration":True,"training_performed":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--train",action="store_true")
    args=parser.parse_args(argv)
    checked=self_test()
    if args.self_test:print(json.dumps(checked));return 0
    if not args.train:parser.error("Use --self-test or --train after root protocol is frozen")
    if REPORTS.exists() or OUTPUT.exists() or any((ROOT/"runs").glob("print_head_*")):
        raise FileExistsError("This protocol permits only one isolated8-epoch configuration")
    if not PROTOCOL.is_file():raise FileNotFoundError("Root must freeze the new experiment protocol before execution")
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    torch.set_num_threads(3)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device("cuda")
    classes,partitions,prints,audit=load_inputs()
    run=ROOT/"runs"/("print_head_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    run.mkdir(parents=True,exist_ok=False);REPORTS.mkdir(parents=True,exist_ok=False)
    frozen_paths=[PROTOCOL,Path(__file__),ROOT/"train_print_classifier.py",ROOT/"train_robust_classifier.py",
                  ROOT/"train_motion_classifier.py",ROOT/"evaluate_robustness_suite.py",ROOT/"rover_motion.py",
                  USER_MANIFEST,SOURCE,ROOT/"data/metadata/manifest.json",ROOT/"data/metadata/classes.json"]
    hashes={str(path):digest(path) for path in frozen_paths}
    hashes.update({str(ROOT/row["path"]):row["sha256"] for row in prints})
    for row in audit:
        hashes[str(ROOT/row["path"])]=row["rgb_sha256"]
        hashes[str(ROOT/row["mask_path"])]=row["mask_sha256"]
    config={"architecture":"mobilenet_v2","classes":classes,"epochs":8,"batch_size":32,"seed":SEED,
            "mean":base.MEAN,"std":base.STD,"resize_short_edge":232,"center_crop":224,
            "square_pad":True,"square_pad_color":[128,128,128],"crop_margin":.12,
            "source_weights":str(SOURCE),"source_weights_sha256":SOURCE_SHA,"print_manifest_sha256":USER_MANIFEST_SHA,
            "feature_extractor":"ALL parameters/buffers frozen byte-identical; BN eval; Dropout eval",
            "trainable_parameters":["classifier.1.weight","classifier.1.bias"],"optimizer":"AdamW","lr":1e-3,"weight_decay":1e-4,
            "sampling":"24 public +8 print, class-balanced replay, same immutable augmentation helpers",
            "distillation":{"direction":"KL(student||teacher)","weight":10.,"temperature":2.,"public_only":True,
                            "teacher_independent_frozen_eval":True,"objective":"CE(all32)+10*T^2*meanKL(student(public24)/T||teacher(public24)/T)"},
            "conditions":list(CONDITIONS),"selection":"Four frozen public guard families then maximum mean public macroF1; else retain lastREJECTED",
            "guard_limits":{"clean_acc_max_loss":.005,"class_condition_recall_max_loss":.03,"condition_f1_max_loss":.01,"total_confident_errors_max_increase":0},
            "user_data_role":"37 train/development ROI, no random split; score fit only after checkpoint selection",
            "user_source_unknown":True,"complete_source_leakage_absence_claimed":False,"test_pixels_used":False,"deployed":False,
            "hashes":hashes,"status":"TRAINING_OFFLINE_ONLY"}
    save(REPORTS/"protocol.json",config);save(REPORTS/"self_test.json",checked);save(run/"config.json",config)
    save(REPORTS/"public_inputs.json",{"files":audit,"test_pixels_read":False})
    print(json.dumps({"stage":"preparing","run":str(run),"epochs":8,"kl_direction":"student||teacher"}),flush=True)
    public_train=base.LeafDataset(partitions["train"],"train")
    mixed=MixedTraining(public_train,prints);batches=MixedBatches(partitions["train"],prints)
    train_loader=DataLoader(ReplayDomains(mixed),batch_sampler=batches,num_workers=0,pin_memory=True)
    assets=base.LeafDataset(partitions["val"],"train")
    valsets={c:Validation(partitions["val"],c,list(zip(assets.originals,assets.leaves))) for c in CONDITIONS}
    loaders={c:DataLoader(ds,batch_size=64,num_workers=0,pin_memory=True) for c,ds in valsets.items()}
    model=base.create_model(classes,SOURCE).to(device)
    teacher=base.create_model(classes,SOURCE).to(device).eval()
    for parameter in teacher.parameters():parameter.requires_grad=False
    if any(a.data_ptr()==b.data_ptr() for a,b in zip(model.parameters(),teacher.parameters())):
        raise RuntimeError("Teacher must be independent, not aliased with the student")
    reference,reference_scores=measure(model,loaders,device,classes)
    save(run/"reference_public.json",reference);save(run/"reference_public_predictions.json",reference_scores)
    initial_features=fingerprint(model.features.state_dict())
    initial_teacher=fingerprint(teacher.state_dict())
    save(run/"frozen_features_before.json",initial_features);save(run/"frozen_teacher_before.json",initial_teacher)
    optimizer=torch.optim.AdamW(frozen_student(model),lr=1e-3,weight_decay=1e-4)
    history=[];best=None;started=time.perf_counter()
    for epoch in range(1,9):
        batches.epoch=epoch;frozen_student(model)
        loss_sum=ce_sum=kl_sum=0.;steps=0
        for images,labels,is_public in train_loader:
            images,labels,is_public=images.to(device),labels.to(device),is_public.to(device)
            if any(layer.training for layer in teacher.modules()):raise RuntimeError("Teacher left eval mode")
            with torch.no_grad():teacher_logits=teacher(images[is_public]).detach()
            optimizer.zero_grad(set_to_none=True)
            loss,ce,kl=objective(model(images),teacher_logits,labels,is_public)
            loss.backward();optimizer.step()
            loss_sum+=float(loss.detach());ce_sum+=float(ce.detach());kl_sum+=float(kl.detach());steps+=1
        features=fingerprint(model.features.state_dict());teacher_state=fingerprint(teacher.state_dict())
        if features!=initial_features or teacher_state!=initial_teacher:raise RuntimeError("Frozen features or teacher changed")
        save(run/f"epoch_{epoch}_frozen_hashes.json",{"features_byte_identical":True,"teacher_byte_identical":True,
                                                    "features":features,"teacher":teacher_state})
        metrics,predictions=measure(model,loaders,device,classes)
        decision=guards(metrics,reference,classes)
        record={"epoch":epoch,"mean_loss":loss_sum/steps,"mean_ce":ce_sum/steps,"mean_kl_student_teacher":kl_sum/steps,
                "steps":steps,"public_samples":steps*24,"print_samples":steps*8,"guards":decision,"metrics":metrics,
                "elapsed_training_seconds":time.perf_counter()-started}
        history.append(record);save(run/"history.json",history)
        save(run/f"epoch_{epoch}_public_predictions.json",predictions);torch.save(model.state_dict(),run/f"epoch_{epoch}_weights.pt")
        if decision["eligible"] and (best is None or decision["mean_public_macro_f1"]>best["guards"]["mean_public_macro_f1"]):
            best=record;torch.save(model.state_dict(),run/"best_weights.pt")
        print(json.dumps({"epoch":epoch,"loss":record["mean_loss"],"ce":record["mean_ce"],"kl":record["mean_kl_student_teacher"],
                          "guards":decision,"elapsed_training_seconds":record["elapsed_training_seconds"]}),flush=True)
    selected=best if best is not None else history[-1]
    checkpoint=run/("best_weights.pt" if best is not None else "epoch_8_weights.pt")
    model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location=device))
    if fingerprint(model.features.state_dict())!=initial_features or fingerprint(teacher.state_dict())!=initial_teacher:
        raise RuntimeError("Selected checkpoint features or frozen teacher changed")
    save(run/"selection_frozen_before_user_fit.json",{"epoch":selected["epoch"],"checkpoint_sha256":digest(checkpoint),
                  "eligible":best is not None,"user_metrics_computed_yet":False,
                  "reason":"Highest eligible mean public macroF1" if best else "None eligible; last epoch retained asREJECTED diagnostic"})
    user_selected=evaluate_user_fit(model,prints,classes,device)
    user_baseline=evaluate_user_fit(teacher,prints,classes,device)
    save(REPORTS/"user_development_fit.json",{"scope":"Training/development fit ONLY, no selection or independence",
                                            "baseline":user_baseline,"candidate":user_selected})
    config.update(selected_epoch=selected["epoch"],checkpoint_sha256=digest(checkpoint),public_guards_passed=best is not None,
                  status="REJECTED" if best is None else "PENDING_EXPORT_PARITY")
    exported=None
    if best is not None:
        staging=run/"onnx_staging"
        try:
            exported=base.export_onnx(model,staging,config,{"clean":loaders["clean"],"challenge":loaders["motion_blur"]})
            selected_scores=json.loads((run/f"epoch_{selected['epoch']}_public_predictions.json").read_text())
            parity=verify_export_decisions(staging/"model.onnx",loaders,selected_scores)
            save(staging/"all_public_decision_parity.json",parity)
            if not parity["passed"]:raise RuntimeError("Public decision export parity failed")
            config["status"]="ELIGIBLE_OFFLINE_ONLY"
            config["onnx_sha256"]=digest(staging/"model.onnx")
            exported["all_public_decision_parity"]=parity;exported["onnx_sha256"]=config["onnx_sha256"]
            save(staging/"config.json",config)  # Final status BEFORE publishing the directory.
            if any(digest(Path(path))!=value for path,value in hashes.items()):
                raise RuntimeError("Frozen source/data changed before export finalization")
            if fingerprint(model.features.state_dict())!=initial_features or fingerprint(teacher.state_dict())!=initial_teacher:
                raise RuntimeError("Frozen features or teacher changed before output publication")
            if not staging.resolve().is_relative_to(run.resolve()) or not OUTPUT.resolve().is_relative_to((ROOT/"models").resolve()):
                raise ValueError("Candidate output escaped its owned workspace")
            if OUTPUT.exists():raise FileExistsError("Never overwrite an existing candidate")
            staging.rename(OUTPUT)
        except Exception as exc:
            config["status"]="REJECTED_EXPORT_PARITY"
            save(staging/"config.json",config)
            exported={"passed":False,"error":type(exc).__name__,"staging":str(staging),"final_output_published":False}
    if any(digest(Path(path))!=value for path,value in hashes.items()):raise RuntimeError("Frozen source/manifest changed")
    report={"created_utc":datetime.now(timezone.utc).isoformat(),"run":str(run),"status":config["status"],"config":config,
            "reference_public":reference,"history":history,"selected_epoch":selected["epoch"],"selected_checkpoint":str(checkpoint),
            "selected_checkpoint_sha256":digest(checkpoint),"public_guards_passed":best is not None,
            "user_fit_used_for_selection":False,"user_fit_baseline":user_baseline["metrics"],"user_fit_selected":user_selected["metrics"],
            "export":exported,"output":str(OUTPUT) if config["status"]=="ELIGIBLE_OFFLINE_ONLY" else None,
            "training_seconds":round(time.perf_counter()-started,3),"test_pixels_used":False,"pi_accessed":False,"deployed":False,
            "limitations":["37 repeated printed-template training/development crops; final fit is not independent accuracy",
                 "Printed source identity unknown, potentially related to public data; no complete leakage exclusion claim",
                 "Six public validation conditions are reused diagnostics, not a new independent holdout",
                 "Original detector/verifier/quality/tracker unchanged; crop fit is not full-pipeline improvement",
                 "Offline eligibility/export does not authorize deployment or demonstrate biological field performance"]}
    save(run/"report.json",report);save(REPORTS/"report.json",report);save(run/"final_config.json",config)
    print(json.dumps({"status":config["status"],"selected_epoch":selected["epoch"],"user_development_fit_before":user_baseline["metrics"]["accuracy"],
                      "user_development_fit_after":user_selected["metrics"]["accuracy"],"output":report["output"],"deployed":False}),flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
