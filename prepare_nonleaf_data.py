"""Build scene-grouped public non-leaf crops from a pinned COCO panoptic subset.

Only JPEG/PNG/JSON archive members are read. No user photographs are used.
Object crops are filtered against annotated vegetation; scene backgrounds are
kept only when almost entirely covered by a known non-vegetation stuff mask.
Every crop from the same source scene has the same split. Multiple crops are
regions of one scene, never independent scenes or augmentation variants.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/nonleaf"
META = ROOT / "data/metadata/nonleaf_manifest.json"
REPORT = ROOT / "nonleaf_dataset_report.json"
REPO = "lightly-ai/coco128_yolo"
COMMIT = "6d6f7a059ea63efffdbe48aca2d52c04d4c510e9"
BASE = f"https://github.com/{REPO}/releases/download/v0.0.2/"
ASSETS = {
    "images.zip": "83f6cf5b28fa705d327b3fc4add1892698734d94e2f965051a4ec5362b8e49dd",
    "panoptic_segmentation.zip": "8200aa8b0cebc92408a9d9a2ccab889a1b62bd0d4aab0d5502debf1bdc036b7f",
}
SEED = 20261022
VEGETATION = {"potted plant", "flower", "tree-merged", "grass-merged"}
AMBIGUOUS_THINGS = {"potted plant", "vase", "broccoli", "carrot", "banana", "apple",
                    "sandwich", "orange", "hot dog", "pizza", "donut", "cake", "bowl",
                    "dining table"}
SAFE_STUFF = {"banner", "blanket", "bridge", "cardboard", "counter", "curtain",
              "door-stuff", "floor-wood", "gravel", "house", "light", "mirror-stuff",
              "net", "pillow", "platform", "railroad", "river", "road", "roof",
              "sand", "sea", "shelf", "snow", "stairs", "tent", "towel", "wall-brick",
              "wall-stone", "wall-tile", "wall-wood", "water-other", "window-blind",
              "window-other", "fence-merged", "ceiling-merged", "sky-other-merged",
              "cabinet-merged", "table-merged", "floor-other-merged", "pavement-merged",
              "mountain-merged", "dirt-merged", "paper-merged", "building-other-merged",
              "rock-merged", "wall-other-merged", "rug-merged"}
# Explicit review exclusions can be added after looking at local contact sheets.
# Keys identify immutable scene+segment+proposal regions, so reruns are reproducible.
VISUAL_REJECTIONS = {key: "Contact-sheet review: incidental vegetation or ambiguous plant-bearing content"
    for key in (
        "000000000724_3813724_whole", "000000000315_6521000_whole",
        "000000000370_6318980_whole", "000000000474_7505287_right_middle",
        "000000001296_6185598_lower", "000000000488_5995922_patch128_1_0.5",
        "000000000488_5995922_patch128_1_1", "000000000488_5995922_patch80_0.5_0.5",
        "000000000488_5995922_patch80_1_0.5", "000000000724_6446682_patch128_0_0",
        "000000000724_6446682_patch128_0_0.5", "000000000724_6446682_patch80_0_0",
        "000000000724_6446682_patch80_0_0.5", "000000000419_5068127_patch128_0.5_0.5",
        "000000000419_5068127_patch128_0_0.5", "000000000419_5068127_patch128_1_0.5",
        "000000000419_5068127_patch80_0_0.5", "000000001000_7446694_patch80_0_0.5",
        "000000008899_5001565_patch192_1_0.5", "000000000071_4475467_patch128_0.5_1",
        "000000000071_4475467_patch128_0_1", "000000000071_4475467_patch128_1_1",
        "000000000071_4475467_patch80_0.5_0.5", "000000000502_10724000_patch128_1_0",
        "000000000502_10724000_patch80_0.5_0", "000000000502_10724000_patch80_1_0",
        "000000000724_15973014_patch80_0.5_0.5", "000000005529_11448498_patch192_1_0",
        "000000000370_3161914_patch80_0.5_0", "000000000370_3161914_patch80_1_0",
        "000000007795_10464177_patch80_0.5_0.5",
    )}
AUDITED_CANDIDATES_SHA256 = "76f55ec3fd71db208be6740118337423d413293fffd411bcadefdbc277f1a389"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str) -> bytes:
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": "rover-public-nonleaf-dataset"}), timeout=45) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def archive(name: str) -> ZipFile:
    path = OUT / "archives" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(fetch(BASE + name))
    if digest(path.read_bytes()) != ASSETS[name]:
        raise RuntimeError(f"Archive SHA256 mismatch: {name}")
    return ZipFile(path)


def source_tree() -> dict:
    path = OUT / "source_tree.json"
    if path.exists():
        tree = json.loads(path.read_text(encoding="utf-8"))
    else:
        tree = json.loads(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"))
        path.write_text(json.dumps(tree, indent=2), encoding="utf-8")
    if tree.get("truncated") or tree.get("sha") != COMMIT:
        raise RuntimeError("Pinned Git tree is incomplete or has a different revision")
    return {item["path"]: item for item in tree["tree"] if item["type"] == "blob"}


def verified_member(z: ZipFile, name: str, tree: dict) -> bytes:
    data = z.read(name)
    item = tree[name]
    actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if actual != item["sha"] or len(data) != item["size"]:
        raise RuntimeError(f"Pinned source Git SHA mismatch: {name}")
    return data


def save_member(name: str, data: bytes) -> Path:
    # Whitelisted source member names are still checked before filesystem writes.
    target = (OUT / "source" / name).resolve()
    base = (OUT / "source").resolve()
    if base not in target.parents:
        raise RuntimeError("Archive member escaped the intended source directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and digest(target.read_bytes()) != digest(data):
        raise RuntimeError(f"Existing source member differs: {name}")
    if not target.exists():
        target.write_bytes(data)
    return target


def box_iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    aa, ba = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    return inter / max(1, aa + ba - inter)


def candidates(segment: dict, category: dict):
    x, y, w, h = segment["bbox"]
    whole = [x, y, x+w, y+h]
    name = category["name"]
    if category["isthing"]:
        yield "whole", whole, .55 if name == "person" else .60
        if name == "person":
            # These are geometric regions, not face/hand identity annotations.
            for tag, fractions in (("upper", (.10, .00, .90, .32)),
                                   ("torso", (.10, .22, .90, .68)),
                                   ("lower", (.10, .58, .90, 1.00)),
                                   ("left_middle", (.00, .32, .48, .76)),
                                   ("right_middle", (.52, .32, 1.00, .76))):
                a,b,c,d=fractions
                yield tag, [x+a*w, y+b*h, x+c*w, y+d*h], .78
        elif min(w,h) >= 128:
            yield "center", [x+.15*w, y+.15*h, x+.85*w, y+.85*h], .85
    else:
        yield "whole_stuff", whole, .98
        # Distinct real patches of the annotated surface, not image variants.
        for side in (192, 128, 80):
            if w < side or h < side:
                continue
            for ry in (0, .5, 1):
                for rx in (0, .5, 1):
                    left,top=x+rx*(w-side),y+ry*(h-side)
                    yield f"patch{side}_{rx}_{ry}", [left,top,left+side,top+side], .98


def load_scenes():
    tree = source_tree()
    scenes, skipped = [], Counter()
    with archive("images.zip") as images, archive("panoptic_segmentation.zip") as panoptic:
        for source_split in ("train2017", "val2017"):
            ann_name = f"panoptic_segmentation/annotations/panoptic_{source_split}.json"
            data = verified_member(panoptic, ann_name, tree)
            save_member(ann_name, data)
            document = json.loads(data)
            categories = {item["id"]: item for item in document["categories"]}
            licenses = {item["id"]: item for item in document["licenses"]}
            image_map = {item["id"]: item for item in document["images"]}
            for annotation in document["annotations"]:
                info = image_map[annotation["image_id"]]
                # Do not make derivative crops from CC NoDerivs photographs.
                if info["license"] in (3,6):
                    skipped["no_derivatives_license"] += 1
                    continue
                image_name=f"images/{source_split}/{info['file_name']}"
                mask_name=f"panoptic_segmentation/annotations/panoptic_{source_split}/{annotation['file_name']}"
                image_bytes=verified_member(images,image_name,tree)
                mask_bytes=verified_member(panoptic,mask_name,tree)
                image_path=save_member(image_name,image_bytes)
                mask_path=save_member(mask_name,mask_bytes)
                with Image.open(BytesIO(image_bytes)) as source:
                    source.load(); image=source.convert("RGB")
                with Image.open(BytesIO(mask_bytes)) as source:
                    source.load(); rgb=np.asarray(source.convert("RGB"),dtype=np.uint32)
                ids=rgb[:,:,0]+256*rgb[:,:,1]+65536*rgb[:,:,2]
                if image.size != (info["width"],info["height"]) or ids.shape != (image.height,image.width):
                    raise RuntimeError("Image/panoptic geometry mismatch")
                segments=annotation["segments_info"]
                # COCO IDs are globally unique across train2017 and val2017.
                pixel_hash=digest(str(image.size).encode()+image.tobytes())
                scenes.append({"id":info["id"],"group_id":f"coco2017:{info['id']}",
                    "source_split":source_split,"info":info,"license":licenses[info["license"]],
                    "image":image,"mask":ids,"segments":segments,"categories":categories,
                    "source_path":image_path.relative_to(ROOT).as_posix(),
                    "mask_path":mask_path.relative_to(ROOT).as_posix(),
                    "source_sha256":digest(image_bytes),"source_pixel_sha256":pixel_hash,
                    "source_git_sha":tree[image_name]["sha"],"mask_sha256":digest(mask_bytes),
                    "has_person":any(categories[s["category_id"]]["name"]=="person" for s in segments)})
    return scenes,skipped


def assign_splits(scenes):
    # Unite byte/pixel duplicates before assigning any crop a split.
    by_pixel=defaultdict(list)
    for scene in scenes:
        by_pixel[scene["source_pixel_sha256"]].append(scene)
    strata=defaultdict(list)
    for group in by_pixel.values():
        group_id=min(scene["group_id"] for scene in group)
        for scene in group:
            scene["group_id"]=group_id
        strata[any(scene["has_person"] for scene in group)].append(group)
    rng=random.Random(SEED)
    for key,groups in sorted(strata.items()):
        groups.sort(key=lambda group:group[0]["group_id"])
        rng.shuffle(groups)
        n_test,n_val=max(1,round(len(groups)*.15)),max(1,round(len(groups)*.15))
        for index,group in enumerate(groups):
            split="test" if index<n_test else "val" if index<n_test+n_val else "train"
            for scene in group:
                scene["split"]=split


def make_crops(scenes):
    rows,rejected=[],Counter()
    seen_pixels={}
    for scene in sorted(scenes,key=lambda s:s["id"]):
        ids=scene["mask"]
        categories=scene["categories"]
        veg_ids=[segment["id"] for segment in scene["segments"]
                 if categories[segment["category_id"]]["name"] in VEGETATION]
        vegetation=np.isin(ids,veg_ids) if veg_ids else np.zeros_like(ids,dtype=bool)
        scene_count=0
        for segment in sorted(scene["segments"],key=lambda s:(categories[s["category_id"]]["name"]!="person",-s["area"])):
            category=categories[segment["category_id"]]
            name=category["name"]
            if segment.get("iscrowd") or name in AMBIGUOUS_THINGS or (not category["isthing"] and name not in SAFE_STUFF):
                rejected["ambiguous_or_not_safe_category"]+=1;continue
            accepted_boxes=[]
            max_segment=5 if name=="person" else 4 if not category["isthing"] else 2
            for proposal,box,min_purity in candidates(segment,category):
                x1,y1,x2,y2=[round(value) for value in box]
                x1,x2=max(0,x1),min(scene["image"].width,x2)
                y1,y2=max(0,y1),min(scene["image"].height,y2)
                box=[x1,y1,x2,y2]
                if min(x2-x1,y2-y1)<48 or (x2-x1)*(y2-y1)<3072:
                    rejected["small_region"]+=1;continue
                if any(box_iou(box,other)>.72 for other in accepted_boxes):
                    rejected["overlapping_same_segment_region"]+=1;continue
                patch=ids[y1:y2,x1:x2]
                purity=float((patch==segment["id"]).mean())
                veg_fraction=float(vegetation[y1:y2,x1:x2].mean())
                unknown_fraction=float((patch==0).mean())
                if purity<min_purity or veg_fraction>0 or unknown_fraction>.02:
                    rejected["purity_vegetation_or_unknown_filter"]+=1;continue
                crop_id=f"{scene['id']:012d}_{segment['id']}_{proposal}"
                crop=scene["image"].crop(box)
                pixel_hash=digest(str(crop.size).encode()+crop.tobytes())
                if pixel_hash in seen_pixels:
                    rejected["exact_duplicate_pixels"]+=1;continue
                seen_pixels[pixel_hash]=scene["group_id"]
                role="person_region" if name=="person" else "object" if category["isthing"] else "stuff_background"
                target=OUT/"crops"/scene["split"]/(crop_id+".jpg")
                target.parent.mkdir(parents=True,exist_ok=True)
                buffer=BytesIO()
                crop.save(buffer,format="JPEG",quality=95,subsampling=0)
                encoded=buffer.getvalue()
                if target.exists():
                    if digest(target.read_bytes())!=digest(encoded):
                        raise RuntimeError("Existing crop differs from its reproducible encoding")
                else:
                    temporary=target.with_suffix(".jpg.tmp")
                    temporary.write_bytes(encoded)
                    temporary.replace(target)
                with Image.open(BytesIO(encoded)) as verified:
                    verified.load()
                    if verified.size!=crop.size:raise RuntimeError("Crop encoding changed geometry")
                pixels=np.asarray(crop,dtype=np.int16)
                green=float(((pixels[:,:,1]>pixels[:,:,0]*1.12)&(pixels[:,:,1]>pixels[:,:,2]*1.10)&(pixels[:,:,1]>45)).mean())
                row={"path":target.relative_to(ROOT).as_posix(),"label":"nonleaf","class_index":0,
                    "is_leaf":False,"split":scene["split"],"group_id":scene["group_id"],
                    "kind":role,"category":name,"semantic_category":name,"semantic_role":role,
                    "crop_id":crop_id,"source_scene_id":scene["id"],"source_split":scene["source_split"],
                    "source_path":scene["source_path"],"source_sha256":scene["source_sha256"],
                    "source_pixel_sha256":scene["source_pixel_sha256"],"source_git_sha":scene["source_git_sha"],
                    "source":f"https://github.com/{REPO}/tree/{COMMIT}",
                    "source_image_url":scene["info"]["coco_url"],"flickr_source_url":scene["info"]["flickr_url"],
                    "license":scene["license"],"crop_xyxy":box,"segment_id":segment["id"],
                    "panoptic_mask_path":scene["mask_path"],"panoptic_mask_sha256":scene["mask_sha256"],
                    "region_purity":purity,"minimum_region_purity":min_purity,
                    "vegetation_fraction":veg_fraction,"unknown_fraction":unknown_fraction,
                    "green_pixel_fraction":green,"can_host_synthetic_leaf":True,
                    "audit_status":"rule_filtered","width":crop.width,"height":crop.height,
                    "sha256":digest(encoded),"pixel_sha256":pixel_hash,"bytes":len(encoded)}
                rows.append(row);accepted_boxes.append(box);scene_count+=1
                if len(accepted_boxes)>=max_segment:break
                if scene_count>=14:break
            if scene_count>=14:break
    return rows,rejected


def audit_sheets(rows):
    # Include every crop in reproducible pages; smaller category samples can be
    # reviewed separately without confusing review coverage with dataset size.
    qa=OUT/"qa";qa.mkdir(parents=True,exist_ok=True)
    ordered=sorted(rows,key=lambda row:(row["kind"],row["category"],row["crop_id"]))
    expected_pages={f"audit_{index:02d}.jpg" for index in range((len(ordered)+47)//48)}
    for old in qa.glob("audit_[0-9][0-9].jpg"):
        if old.name not in expected_pages:
            # Archive obsolete generated pages instead of leaving them in the
            # accepted-crop preview glob after a visual exclusion shrinks it.
            retired=qa/"retired"/old.name
            retired.parent.mkdir(parents=True,exist_ok=True)
            old.replace(retired)
    for page,start in enumerate(range(0,len(ordered),48)):
        subset=ordered[start:start+48]
        sheet=Image.new("RGB",(8*160,6*188),"white");draw=ImageDraw.Draw(sheet)
        for index,row in enumerate(subset):
            x,y=(index%8)*160,(index//8)*188
            with Image.open(ROOT/row["path"]) as image:
                image=ImageOps.contain(image.convert("RGB"),(156,156))
                sheet.paste(image,(x+(160-image.width)//2,y+28+(156-image.height)//2))
            draw.text((x+2,y+1),f"{start+index}: {row['category'][:17]}",fill="black")
            draw.text((x+2,y+14),f"{row['split']} p{row['region_purity']:.2f}",fill="black")
        sheet.save(qa/f"audit_{page:02d}.jpg",quality=92)
    (qa/"audit_order.json").write_text(json.dumps([row["crop_id"] for row in ordered],indent=2),encoding="utf-8")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    scenes,source_skips=load_scenes()
    assign_splits(scenes)
    rows,rejections=make_crops(scenes)
    if not rows:raise RuntimeError("No safe non-leaf crop candidates")
    # Review occurs after choosing regions, so rejecting one cannot silently add
    # a replacement patch that was never inspected in the original contact set.
    candidate_signature=digest("\n".join(sorted(row["crop_id"] for row in rows)).encode())
    if candidate_signature!=AUDITED_CANDIDATES_SHA256:
        raise RuntimeError("Crop proposals changed since visual review; audit again before publishing a manifest")
    reviewed_candidates=len(rows)
    accepted=[]
    for row in rows:
        if row["crop_id"] in VISUAL_REJECTIONS:
            source=(ROOT/row["path"]).resolve()
            target=(OUT/"rejected"/source.name).resolve()
            if OUT.resolve() not in source.parents or OUT.resolve() not in target.parents:
                raise RuntimeError("Review exclusion escaped the intended nonleaf directory")
            target.parent.mkdir(parents=True,exist_ok=True)
            source.replace(target)
            rejections["visual_review_exclusion"]+=1
        else:
            row["audit_status"]="visually_reviewed_contact_sheet"
            accepted.append(row)
    rows=accepted
    group_sets={split:{row["group_id"] for row in rows if row["split"]==split} for split in ("train","val","test")}
    source_sets={split:{row["source_pixel_sha256"] for row in rows if row["split"]==split} for split in group_sets}
    for a,b in (("train","val"),("train","test"),("val","test")):
        if group_sets[a]&group_sets[b] or source_sets[a]&source_sets[b]:
            raise RuntimeError("Source scene leakage across non-leaf splits")
    META.parent.mkdir(parents=True,exist_ok=True)
    temporary=META.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(rows,indent=2),encoding="utf-8")
    temporary.replace(META)
    audit_sheets(rows)
    report={"source_repository":f"https://github.com/{REPO}","source_commit":COMMIT,
        "source_documentation":"https://github.com/lightly-ai/coco128_yolo#provenance",
        "original_dataset":"https://cocodataset.org/", "terms":"https://cocodataset.org/#termsofuse",
        "archive_urls_and_sha256":{BASE+name:value for name,value in ASSETS.items()},
        "manifest_path":META.relative_to(ROOT).as_posix(),"manifest_format":"JSON list of row objects",
        "manifest_sha256":digest(META.read_bytes()),"random_seed":SEED,
        "source_annotated_scenes_available":200,"source_scenes_after_license_filter":len(scenes),
        "source_scenes_with_accepted_crops":len({row["group_id"] for row in rows}),
        "source_skips":dict(source_skips),"crop_count":len(rows),
        "crop_splits":dict(Counter(row["split"] for row in rows)),
        "scene_splits":{split:len(groups) for split,groups in group_sets.items()},
        "categories":dict(Counter(row["category"] for row in rows)),
        "kinds":dict(Counter(row["kind"] for row in rows)),
        "person_green_regions_gt_10percent":sum(row["kind"]=="person_region" and row["green_pixel_fraction"]>.10 for row in rows),
        "region_purity_min":min(row["region_purity"] for row in rows),
        "vegetation_fraction_max":max(row["vegetation_fraction"] for row in rows),
        "unknown_fraction_max":max(row["unknown_fraction"] for row in rows),
        "cross_split_source_scene_or_pixel_overlap":0,"duplicate_crop_pixels":0,
        "rejection_counts":dict(rejections),"visual_rejection_list":VISUAL_REJECTIONS,
        "visual_reviewed_candidates":reviewed_candidates,
        "visual_review_rejected_regions":len(VISUAL_REJECTIONS),
        "visual_review_accepted_regions":len(rows),
        "visual_review_candidate_id_set_sha256":candidate_signature,
        "visual_review_method":"All 546 candidate crops inspected in 12 local contact sheets at up to 156x156 per crop; 31 ambiguous or vegetation-containing regions excluded. This is visual QA, not exhaustive pixel-level leaf annotation.",
        "visual_audit_pages":"data/nonleaf/qa/audit_*.jpg",
        "licensing":"Each row retains the original COCO/Flickr image license and attribution source URL. CC NoDerivs sources excluded. Remaining licenses include NonCommercial/ShareAlike restrictions; dataset is for local research, not a blanket CC-BY image collection.",
        "positive_compositing_policy":"Use these same crop images for positive leaf composites, including person regions, with matching scene split AND leaf-group split. A person holding a visible leaf must remain a positive example; never add a universal person veto.",
        "limitations":["This is a small source-scene dataset; crops and variants are not independent scenes",
            "No annotated vegetation in accepted crops does not exclude every incidental unannotated leaf",
            "Person subregions are geometric crops, not verified face or hand annotations",
            "Some categories may be rare or missing; report actual counts",
            "This dataset does not establish field or printed-leaf accuracy"],
        "training_performed":False,"user_images_included":False}
    REPORT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
