"""Isolated motion-augmentation candidate; validation only, no test/camera use.

Hypothesis: linear exposure blur is absent from the first robust fine tune
(which only uses mild Gaussian blur). Add linear blur and small occlusions,
starting from the deployed robust weights, with validation guards against that
deployed model across all six existing diagnostic conditions.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import train_robust_classifier as base
from evaluate_robustness_suite import CONDITIONS, compose_condition, motion_blur, classification_metrics

ROOT=Path(__file__).resolve().parent
SEED=2026102202


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


class MotionTrain(base.LeafDataset):
    def __getitem__(self,index):
        image=self.originals[index].copy()
        synthetic=random.random()>=.45
        if synthetic:
            image=base.compose(self.leaves[index],random)
        elif random.random()<.5:
            image=base.ImageOps.mirror(image)
        image=base.photometric(image,random,synthetic)
        if random.random()<.55:
            image=motion_blur(image,random.choice((3,5,7,9,11)),random.choice(('h','v','d','a')))
        # The object stays visible. The same occlusion distribution applies to
        # all labels and removes at most 5% of the rectangular image area.
        if synthetic and random.random()<.25:
            w=max(1,round(image.width*random.uniform(.10,.20)))
            h=max(1,round(image.height*random.uniform(.10,.25)))
            x=random.randrange(max(1,image.width-w+1))
            y=random.randrange(max(1,image.height-h+1))
            color=random.choice(((118,112,98),(128,128,128),(67,90,44)))
            ImageDraw.Draw(image).rectangle((x,y,x+w-1,y+h-1),fill=color)
        return self.transform(image),self.rows[index]['class_index']


class Validation(Dataset):
    def __init__(self,rows,mode,assets):
        self.rows=rows
        self.tensors=[]
        for row,(rgb,asset) in zip(rows,assets):
            image,_=compose_condition(rgb,asset,row,mode)
            self.tensors.append(base.transforms.Compose([
                base.transforms.Resize(232),base.transforms.CenterCrop(224),
                base.transforms.ToTensor(),base.transforms.Normalize(base.MEAN,base.STD)])(image))
    def __len__(self): return len(self.rows)
    def __getitem__(self,index): return self.tensors[index],self.rows[index]['class_index']


def measure(model,loaders,device,classes):
    model.eval()
    metrics,predictions={},{}
    with torch.inference_mode():
        for condition,loader in loaders.items():
            scores=[]
            for images,_ in loader:
                scores.append(model(images.to(device)).softmax(1).cpu().numpy())
            scores=np.concatenate(scores)
            metrics[condition]=classification_metrics(loader.dataset.rows,scores,classes)
            metrics[condition]['leaf_group_level']=base.summarize(loader.dataset.rows,scores,classes)['leaf_group_level']
            predictions[condition]=scores.tolist()
    return metrics,predictions


def guards(candidate,reference,classes):
    clean=candidate['clean']['accuracy']>=reference['clean']['accuracy']-.005
    per_class=all(candidate[c]['per_class'][label]['recall']>=reference[c]['per_class'][label]['recall']-.03
                  for c in CONDITIONS for label in classes)
    macro=all(candidate[c]['macro_f1']>=reference[c]['macro_f1']-.01 for c in CONDITIONS)
    confident=sum(candidate[c]['high_confidence_errors'] for c in CONDITIONS)<=sum(reference[c]['high_confidence_errors'] for c in CONDITIONS)
    score=float(np.mean([candidate[c]['macro_f1'] for c in CONDITIONS]))
    refscore=float(np.mean([reference[c]['macro_f1'] for c in CONDITIONS]))
    late='Potato___Late_blight'
    target=(candidate['motion_blur']['per_class'][late]['recall']>=reference['motion_blur']['per_class'][late]['recall']+.02)
    return {'clean_accuracy':clean,'per_class_all_conditions':per_class,'macro_f1_each_condition':macro,
            'confident_errors_total':confident,'motion_late_recall_improves_2p':target,
            'score':score,'reference_score':refscore,
            'eligible':bool(clean and per_class and macro and confident and target and score>=refscore+.003)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=6)
    parser.add_argument('--batch-size',type=int,default=32)
    args=parser.parse_args()
    torch.set_num_threads(3)
    assert torch.cuda.is_available()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False
    device=torch.device('cuda')
    manifest_path=ROOT/'data/metadata/manifest.json'
    rows=json.loads(manifest_path.read_text())
    classes=json.loads((ROOT/'data/metadata/classes.json').read_text())
    partitions={split:[r for r in rows if r['split']==split] for split in ('train','val','test')}
    for field in ('group_id','sha256','pixel_sha256'):
        keys={s:{r[field] for r in rs} for s,rs in partitions.items()}
        assert not keys['train']&keys['val'] and not keys['train']&keys['test'] and not keys['val']&keys['test']
    source=ROOT/'runs/robust_20261021_232142/best_weights.pt'
    run=ROOT/'runs'/('motion_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    run.mkdir(parents=True,exist_ok=False)
    config={'architecture':'mobilenet_v2','classes':classes,'seed':SEED,'mean':base.MEAN,'std':base.STD,
            'resize_short_edge':232,'center_crop':224,'square_pad':True,'square_pad_color':[128,128,128],
            'crop_margin':.12,'source_weights':str(source),'source_weights_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'training':'45% original/55% composed, existing photometric;55%linear motion blur3-11,25%composed small occlusion;class balanced;teacher only when correctconfident',
            'selection':'All six reused validation conditions only, never test/user; fixed guards before training in source',
            'test_used':False,'user_camera_images_used':False,'deployed':False,'epochs':args.epochs}
    save(run/'config.json',config)
    print(json.dumps({'run':str(run),'stage':'preparing','config':config}),flush=True)
    train=MotionTrain(partitions['train'],'train')
    # Reuse the already audited alpha geometry, never the segmented JPEG RGB.
    validation_assets=[]
    validation_clean=base.LeafDataset(partitions['val'],'train')
    validation_assets=list(zip(validation_clean.originals,validation_clean.leaves))
    valsets={c:Validation(partitions['val'],c,validation_assets) for c in CONDITIONS}
    loaders={c:DataLoader(ds,batch_size=64,num_workers=0,pin_memory=True) for c,ds in valsets.items()}
    counts=Counter(r['class_index'] for r in train.rows)
    sampler=WeightedRandomSampler([1/counts[r['class_index']] for r in train.rows],len(train),replacement=True,
                                  generator=torch.Generator().manual_seed(SEED))
    train_loader=DataLoader(train,batch_size=args.batch_size,num_workers=0,pin_memory=True,sampler=sampler)
    model=base.create_model(classes,source).to(device)
    teacher=base.create_model(classes,source).to(device).eval()
    for p in teacher.parameters(): p.requires_grad=False
    reference,refpred=measure(model,loaders,device,classes)
    save(run/'reference.json',reference); save(run/'reference_predictions.json',refpred)
    for p in model.parameters(): p.requires_grad=False
    for p in model.features[-5:].parameters(): p.requires_grad=True
    for p in model.classifier.parameters(): p.requires_grad=True
    optimizer=torch.optim.AdamW([{'params':model.features[-5:].parameters(),'lr':5e-5},
                                {'params':model.classifier.parameters(),'lr':1e-4}],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=1e-5)
    scaler=torch.amp.GradScaler('cuda')
    best=None; history=[]; started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train()
        for layer in model.modules():
            if isinstance(layer,nn.BatchNorm2d): layer.eval()
        loss_sum=0; batches=0
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(): teacher_logits=teacher(images)
            with torch.amp.autocast('cuda'):
                logits=model(images)
                loss=F.cross_entropy(logits,labels)
                teacher_scores=teacher_logits.softmax(1)
                eligible=(teacher_scores.max(1).values>.9)&(teacher_scores.argmax(1)==labels)
                if eligible.any():
                    loss=loss+.15*4*F.kl_div(F.log_softmax(logits[eligible]/2,dim=1),
                                             F.softmax(teacher_logits[eligible]/2,dim=1),reduction='batchmean')
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum+=loss.item(); batches+=1
        result,predictions=measure(model,loaders,device,classes)
        promotion=guards(result,reference,classes)
        record={'epoch':epoch,'loss':loss_sum/batches,'guards':promotion,'metrics':result,
                'elapsed_seconds':time.perf_counter()-started}
        history.append(record); save(run/'history.json',history)
        if promotion['eligible'] and (best is None or promotion['score']>best['guards']['score']):
            best=record
            torch.save(model.state_dict(),run/'best_weights.pt')
            save(run/'best_predictions.json',predictions); save(run/'best_validation.json',record)
        scheduler.step()
        print(json.dumps({'epoch':epoch,'loss':record['loss'],'guards':promotion,'elapsed_seconds':record['elapsed_seconds']}),flush=True)
    report={'run':str(run),'created_at_utc':datetime.now(timezone.utc).isoformat(), 'reference':reference,
            'selected':best,'passed':best is not None,'deployed':False,'test_used':False,'user_camera_images_used':False,
            'history':history,'limits':['Reused validation conditions; not independent field evidence',
            'Only six healthy validation leaf groups','Synthetic lineblur/occlusion; real moving-car video still needed']}
    if best:
        model.load_state_dict(torch.load(run/'best_weights.pt',weights_only=True,map_location=device))
        output=ROOT/'models/potato-motion-candidate'
        report['export']=base.export_onnx(model,output,config,{'clean':loaders['clean'],'challenge':loaders['motion_blur']})
        report['output']=str(output)
    save(run/'report.json',report)
    save(ROOT/'reports/motion_classifier_validation.json',report)
    print(json.dumps({'run':str(run),'passed':report['passed'],'output':report.get('output'),'deployed':False}),flush=True)


if __name__=='__main__': main()
