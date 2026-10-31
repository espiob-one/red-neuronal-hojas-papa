"""Train leaf/no-leaf verification separately from disease classification.

Public negatives are grouped by original scene. PlantVillage positives retain
their leaf splits. Camera pictures from the user are never read by this script.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from train_robust_classifier import foreground, photometric, compose, save_json

ROOT = Path(__file__).resolve().parent
SEED = 20261023
TRANSFORM = transforms.Compose([transforms.Resize(232), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])


def stable_seed(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:12],16)


def composite_on_scene(leaf, scene, rng):
    leaf=leaf.copy()
    if rng.random()<.5:
        leaf=ImageOps.mirror(leaf)
    leaf=leaf.rotate(rng.uniform(-35,35),Image.Resampling.BICUBIC,expand=True)
    bbox=leaf.getchannel('A').getbbox()
    if not bbox:
        raise ValueError('Empty alpha')
    leaf=leaf.crop(bbox)
    side=max(leaf.size)
    ratio=rng.uniform(.70,.84)*224/side
    leaf=leaf.resize((max(1,round(leaf.width*ratio)),max(1,round(leaf.height*ratio))),Image.Resampling.BILINEAR)
    scene=ImageOps.fit(scene,(224,224),method=Image.Resampling.BILINEAR).copy()
    scene.paste(leaf,((224-leaf.width)//2,(224-leaf.height)//2),leaf)
    return scene


class VerificationDataset(Dataset):
    """One scene per negative; original and synthetic views share group IDs."""
    def __init__(self, positive_rows, negative_rows, mode):
        self.positive_rows,self.negative_rows,self.mode=positive_rows,negative_rows,mode
        self.originals=[];self.leaves=[];self.scenes=[]
        for row in positive_rows:
            with Image.open(ROOT/row['path']) as source:
                original=source.convert('RGB')
            mask_path=ROOT/'data/segmented'/row['label']/(Path(row['path']).stem+'_final_masked.jpg')
            with Image.open(mask_path) as segmented:
                self.leaves.append(foreground(segmented.convert('RGB'),original))
            self.originals.append(original)
        for row in negative_rows:
            with Image.open(ROOT/row['path']) as source:
                self.scenes.append(source.convert('RGB'))
        self.records=[]
        for i,row in enumerate(positive_rows):
            views=['mixed'] if mode=='train' else ['clean','scene','capture']
            for view in views:
                self.records.append({'index':i,'label':1,'group_id':row['group_id'],'view':view,'path':row['path'],'disease_class':row['label']})
        for i,row in enumerate(negative_rows):
            self.records.append({'index':i,'label':0,'group_id':row['group_id'],'view':'negative','path':row['path'],'kind':row.get('kind',row.get('category','object'))})
        # Freeze all validation/test pixels; no epoch-dependent difficulty.
        self.fixed=None
        if mode!='train':
            self.fixed=[self.render(i) for i in range(len(self.records))]

    def __len__(self):
        return len(self.records)

    def render(self,index):
        record=self.records[index]
        rng=random if self.mode=='train' else random.Random(stable_seed(str(SEED)+record['path']+record['view']))
        i=record['index']
        if record['label']==1:
            choice=rng.random()
            view=record['view']
            if view=='clean' or view=='mixed' and choice<.3:
                image=self.originals[i].copy()
            elif view=='capture' or view=='mixed' and choice<.6:
                image=compose(self.leaves[i],rng)
                image=photometric(image,rng,strong=True)
            else:
                background=self.scenes[rng.randrange(len(self.scenes))]
                image=composite_on_scene(self.leaves[i],background,rng)
                image=photometric(image,rng,strong=True)
        else:
            image=ImageOps.pad(self.scenes[i],(max(self.scenes[i].size),)*2,color=(128,128,128))
            if self.mode=='train':
                if rng.random()<.5:
                    image=ImageOps.mirror(image)
                image=photometric(image,rng,strong=True)
        return TRANSFORM(image),record['label']

    def __getitem__(self,index):
        return self.fixed[index] if self.fixed is not None else self.render(index)


def wilson(k,n):
    if not n:
        return [None,None]
    z=1.96;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(max(0,mid-half)),float(min(1,mid+half))]


def threshold_metrics(records, probabilities, threshold):
    pos=np.array([r['label']==1 for r in records]);pred=probabilities>=threshold
    tp=int((pred&pos).sum());fp=int((pred&~pos).sum())
    grouped=defaultdict(list)
    for record,accepted in zip(records,pred):
        if record['label']==0:
            grouped[record['group_id']].append(bool(accepted))
    bad_scenes=sum(any(items) for items in grouped.values())
    disease={}
    for label in sorted({r.get('disease_class') for r in records if r['label']==1}):
        indices=[i for i,r in enumerate(records) if r.get('disease_class')==label]
        disease[label]=float(pred[indices].mean())
    return {'threshold':float(threshold),'leaf_recall':tp/max(1,int(pos.sum())),
        'negative_false_accept_rate':fp/max(1,int((~pos).sum())),
        'leaf_true_accepts':tp,'leaf_views':int(pos.sum()),'negative_false_accepts':fp,
        'negative_crops':int((~pos).sum()),'negative_scenes':len(grouped),
        'negative_scenes_with_false_accept':int(bad_scenes),
        'negative_scene_false_accept_rate':bad_scenes/max(1,len(grouped)),
        'negative_scene_false_accept_95ci':wilson(bad_scenes,len(grouped)),
        'per_disease_leaf_recall':disease}


def choose_threshold(records, probabilities):
    rows=[threshold_metrics(records,probabilities,t / 100) for t in range(5,96)]
    eligible=[r for r in rows if r['leaf_recall']>=.97 and r['negative_false_accept_rate']<=.05
              and r['negative_scene_false_accept_rate']<=.10
              and min(r['per_disease_leaf_recall'].values())>=.95]
    # Among eligible thresholds, minimize false accepts without sacrificing the
    # predefined recall requirements. Do not consult the held-out test here.
    # A plateau must not silently select the first/lowest threshold. Break an
    # exact validation-metric tie toward the highest eligible threshold; all
    # existing recall/false-accept guards remain unchanged.
    return (min(eligible,key=lambda r:(r['negative_scene_false_accept_rate'],r['negative_false_accept_rate'],
                                      -r['leaf_recall'],-r['threshold'])) if eligible else None),rows


def recalibrate_saved_validation(run):
    """Mechanical tie-policy correction; never read held-out image pixels."""
    records=json.loads((run/'validation_predictions.json').read_text(encoding='utf-8'))
    positives=json.loads((ROOT/'data/metadata/manifest.json').read_text(encoding='utf-8-sig'))
    negatives=json.loads((ROOT/'data/metadata/nonleaf_manifest.json').read_text(encoding='utf-8-sig'))
    allowed={r['group_id'] for r in positives+negatives if r['split']=='val'}
    if not records or any(r['group_id'] not in allowed for r in records):
        raise ValueError('Saved calibration rows must belong exclusively to validation groups')
    probabilities=np.asarray([r['leaf_score'] for r in records])
    if not np.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any():
        raise ValueError('Invalid saved validation probabilities')
    selected,grid=choose_threshold(records,probabilities)
    if selected is None:
        raise RuntimeError('No threshold satisfies the existing validation guards')
    output=ROOT/'models/leaf-verifier-candidate'
    config=json.loads((output/'config.json').read_text())
    report=json.loads((run/'verifier_report.json').read_text())
    if config['source_run'] != str(run):
        raise ValueError('Candidate configuration does not reference the requested frozen run')
    archive=ROOT/'reports'/('verifier_policy_correction_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    archive.mkdir(parents=True,exist_ok=False)
    save_json(archive/'previous_config.json',config)
    save_json(archive/'previous_verifier_report.json',report)
    previous_selected=report['selected']
    correction={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'reason':'Exact metric ties previously inherited the ascending grid order (.05); add conservative highest-threshold tie break.',
        'policy':'Minimize validation scene/crop false accepts, maximize validation recall, then highest eligible threshold.',
        'previous_threshold':config['threshold'],'threshold':selected['threshold'],
        'epoch':previous_selected['epoch'],'validation':selected,
        'validation_predictions_sha256':hashlib.sha256((run/'validation_predictions.json').read_bytes()).hexdigest(),
        'model_sha256':hashlib.sha256((output/'model.onnx').read_bytes()).hexdigest(),
        'weights_sha256':hashlib.sha256((run/'best_weights.pt').read_bytes()).hexdigest(),
        'test_or_external_data_used_for_recalibration':False,'retraining':False,'archive':str(archive)}
    config.update(threshold=selected['threshold'],validation=selected,threshold_policy_correction=correction)
    if 'heldout_test' in config:
        config['heldout_test_previous_policy']=config.pop('heldout_test')
    report['selected_before_threshold_policy_correction']=previous_selected
    report['selected']={**previous_selected,'selected':selected}
    report['threshold_policy_correction']=correction
    if 'heldout_test' in report:
        report['heldout_test_previous_policy']=report.pop('heldout_test')
    save_json(archive/'recalibrated_validation_threshold_grid.json',grid)
    save_json(archive/'frozen_threshold.json',correction)
    save_json(output/'config.json',config)
    save_json(run/'verifier_report.json',report)
    save_json(ROOT/'verifier_report.json',report)
    # Hash the exact configuration only after it is saved; avoid self-hashing.
    correction={**correction,'config_sha256':hashlib.sha256((output/'config.json').read_bytes()).hexdigest()}
    save_json(archive/'frozen_threshold.json',correction)
    print(json.dumps(correction,indent=2),flush=True)


def evaluate(model,loader,device):
    model.eval();values=[]
    with torch.inference_mode():
        for tensors,_ in loader:
            values.extend(model(tensors.to(device)).softmax(1)[:,1].cpu().tolist())
    return np.asarray(values)


def audit(positives,negatives):
    for family,rows in [('leaf',positives),('negative',negatives)]:
        for key in ['group_id','sha256']:
            sets={s:{r[key] for r in rows if r['split']==s} for s in ['train','val','test']}
            if sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test']:
                raise ValueError(f'{family} split overlap in {key}')
        if any(sum(r['split']==s for r in rows)==0 for s in ['train','val','test']):
            raise ValueError(f'Missing {family} split')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=6)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--recalibrate-saved-validation',type=Path)
    args=parser.parse_args()
    if args.recalibrate_saved_validation:
        recalibrate_saved_validation(args.recalibrate_saved_validation.resolve())
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires the PC CUDA GPU')
    torch.set_num_threads(4);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    positives=json.loads((ROOT/'data/metadata/manifest.json').read_text(encoding='utf-8-sig'))
    negative_path=ROOT/'data/metadata/nonleaf_manifest.json'
    negatives=json.loads(negative_path.read_text(encoding='utf-8-sig'))
    audit(positives,negatives)
    run=ROOT/'runs'/('verifier_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir()
    datasets={}
    for split in ['train','val']:
        datasets[split]=VerificationDataset([r for r in positives if r['split']==split],[r for r in negatives if r['split']==split],split)
    records=datasets['train'].records
    counts={label:sum(r['label']==label for r in records) for label in [0,1]}
    sampler=torch.utils.data.WeightedRandomSampler([1/counts[r['label']] for r in records],len(records),replacement=True,generator=torch.Generator().manual_seed(SEED))
    loaders={s:DataLoader(ds,batch_size=args.batch_size,num_workers=0,sampler=sampler if s=='train' else None,pin_memory=True) for s,ds in datasets.items()}
    source=ROOT/'runs/robust_20261021_232142/best_weights.pt'
    weights=torch.load(source,map_location='cpu',weights_only=True)
    model=models.mobilenet_v2(weights=None)
    model.features.load_state_dict({k[len('features.'):]:v for k,v in weights.items() if k.startswith('features.')})
    model.classifier[1]=nn.Linear(model.last_channel,2)
    for parameter in model.features.parameters():
        parameter.requires_grad=False
    for parameter in model.features[-5:].parameters():
        parameter.requires_grad=True
    device='cuda';model.to(device)
    optimizer=torch.optim.AdamW([{'params':model.features[-5:].parameters(),'lr':1e-4},{'params':model.classifier.parameters(),'lr':4e-4}],weight_decay=1e-4)
    criterion=nn.CrossEntropyLoss();scaler=torch.amp.GradScaler('cuda')
    history=[];best=None;best_key=None
    started=time.monotonic()
    for epoch in range(1,args.epochs+1):
        model.train()
        for module in model.modules():
            if isinstance(module,nn.BatchNorm2d):module.eval()
        losses=[]
        for images,labels in loaders['train']:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                loss=criterion(model(images.to(device)),labels.to(device))
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();losses.append(loss.item())
        probabilities=evaluate(model,loaders['val'],device)
        selected,grid=choose_threshold(datasets['val'].records,probabilities)
        record={'epoch':epoch,'train_loss':float(np.mean(losses)),'selected':selected,'elapsed_seconds':round(time.monotonic()-started,1)}
        history.append(record);save_json(run/'history.json',history)
        print(json.dumps(record),flush=True)
        if selected is not None:
            key=(selected['negative_scene_false_accept_rate'],selected['negative_false_accept_rate'],-selected['leaf_recall'])
            if best_key is None or key<best_key:
                best_key=key;best=record;torch.save(model.state_dict(),run/'best_weights.pt')
                save_json(run/'validation_threshold_grid.json',grid)
                save_json(run/'validation_predictions.json',[{**r,'leaf_score':float(p)} for r,p in zip(datasets['val'].records,probabilities)])
    report={'run':str(run),'selected':best,'history':history,'passed_validation':best is not None,'deployed':False,
        'test_used_for_selection':False,'user_camera_images_used':False,
        'negative_manifest_sha256':hashlib.sha256(negative_path.read_bytes()).hexdigest(),
        'limitations':['Public scene negatives and synthetic leaf composites; not field validation.','Crops and multiple views from a scene/leaf are correlated.','Scores are not calibrated open-world probabilities.']}
    if best is not None:
        # Freeze the chosen epoch/threshold before reading any held-out pixels.
        save_json(run/'frozen_selection.json',best)
        model.load_state_dict(torch.load(run/'best_weights.pt',map_location=device,weights_only=True))
        test=VerificationDataset([r for r in positives if r['split']=='test'],[r for r in negatives if r['split']=='test'],'test')
        test_loader=DataLoader(test,batch_size=args.batch_size,num_workers=0)
        test_probabilities=evaluate(model,test_loader,device)
        report['heldout_test']=threshold_metrics(test.records,test_probabilities,best['selected']['threshold'])
        output=ROOT/'models/leaf-verifier-candidate';output.mkdir(exist_ok=True)
        model.cpu().eval();dummy=torch.zeros(1,3,224,224)
        torch.onnx.export(model,dummy,output/'model.onnx',input_names=['image'],output_names=['logits'],opset_version=17,dynamo=False)
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=1
        session=ort.InferenceSession(str(output/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        errors=[];matches=[]
        for i in np.linspace(0,len(test)-1,20,dtype=int):
            tensor=test[int(i)][0].unsqueeze(0)
            with torch.inference_mode():reference=model(tensor).numpy()
            result=session.run(None,{'image':tensor.numpy()})[0]
            errors.append(float(np.max(np.abs(reference-result))));matches.append(bool(reference.argmax()==result.argmax()))
        report['export_validation']={'max_logit_error':max(errors),'all_argmax_match':all(matches),'comparisons':len(errors)}
        if max(errors)>1e-4 or not all(matches):raise RuntimeError('Verifier ONNX export mismatch')
        save_json(output/'config.json',{'classes':['no_leaf','leaf'],'threshold':best['selected']['threshold'],'mean':[.485,.456,.406],
            'std':[.229,.224,.225],'resize_short_edge':232,'center_crop':224,'source_run':str(run),
            'scope':'Experimental binary verifier; not a potato species identifier','validation':best['selected'],'heldout_test':report['heldout_test']})
        report['output']=str(output)
    save_json(run/'verifier_report.json',report);save_json(ROOT/'verifier_report.json',report)
    print(json.dumps({'passed_validation':report['passed_validation'],'output':report.get('output'),'heldout_test':report.get('heldout_test')}),flush=True)


if __name__=='__main__':main()
