"""Build a frozen, grouped, synthetic potato-leaf YOLO detection dataset.

No model imports, downloads, test pixels, camera images or training occur here.
Only already-reviewed COCO non-leaf crops are used as backgrounds. PlantVillage
segmented JPEGs supply alpha geometry; original photographs supply leaf RGB.
Run normally to generate/audit. After actually inspecting every listed contact
sheet, use --review-visual with a concrete review note to authorize local training.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "data/potato_detection_v1"
DEFAULT_REPORT = ROOT / "reports/potato_detection_data_v1"
SEED = 2026102207
SIZE = (640, 360)
SPLITS = ("train", "val")
VISUAL_BACKGROUND_EXCLUSIONS = {
    "coco2017:86": "Full-crop review: forest ground contains fallen leaves despite panoptic dirt/vegetation0",
    "coco2017:3934": "Full-crop review: flower lei/decoration has ambiguous green plant-shaped pieces",
}
VISUAL_BACKGROUND_REVIEW = "reports/potato_detection_data_v1_visual_rejection/evidence/background_source_qa_review.json"
PLANT_SOURCE = {"repository": "https://github.com/spMohanty/PlantVillage-Dataset",
                "commit": "7f7ecc7e1eaca78107e3affe7cb5abd9427e139a",
                "license": {"name": "Not explicitly stated in the pinned repository README",
                            "upstream_statement": "open access",
                            "url": "https://github.com/spMohanty/PlantVillage-Dataset/blob/7f7ecc7e1eaca78107e3affe7cb5abd9427e139a/README.md"}}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def foreground(segmented, original):
    """Remove boundary-connected black; retain enclosed dark disease tissue."""
    if segmented.size != original.size:
        raise ValueError("RGB/alpha source geometry differs")
    mask = Image.fromarray((np.asarray(segmented.convert("RGB")).max(2) > 20).astype(np.uint8)).copy()
    w, h = mask.size
    access = mask.load()
    for xy in [(x, y) for x in range(w) for y in (0, h - 1)] + [(x, y) for y in range(h) for x in (0, w - 1)]:
        if access[xy] == 0:
            ImageDraw.floodfill(mask, xy, 2, thresh=0)
    alpha = Image.fromarray(np.uint8(np.asarray(mask) != 2) * 255)
    fraction = float((np.asarray(alpha) > 0).mean())
    if not .08 <= fraction <= .95:
        raise ValueError(f"Implausible source alpha fraction {fraction}")
    rgba = original.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba.crop(alpha.getbbox()), fraction


def excluded_background_groups(report_root):
    excluded, evidence = set(VISUAL_BACKGROUND_EXCLUSIONS), []
    visual = ROOT / VISUAL_BACKGROUND_REVIEW
    if visual.exists():
        evidence.append({"path": relative(visual), "sha256": digest(visual.read_bytes())})
    for path in sorted(Path(report_root).glob("hard_negative_mining*/derivative_qa_review.json")):
        for row in read_json(path).get("rows", []):
            if row.get("can_train") is False:
                excluded.add(row["group_id"])
        evidence.append({"path": relative(path), "sha256": digest(path.read_bytes())})
    return excluded, evidence


def background_eligible(row, excluded):
    return (row.get("split") in SPLITS and row.get("is_leaf") is False
            and row.get("vegetation_fraction") == 0
            and row.get("unknown_fraction", 1) <= .02
            and row.get("region_purity", 0) >= .90
            and row.get("can_host_synthetic_leaf") is True
            and row.get("audit_status", "").startswith("visually_reviewed")
            and row.get("group_id") not in excluded)


def split_collisions(rows, key):
    seen = defaultdict(set)
    for row in rows:
        if row.get(key):
            seen[row[key]].add(row["split"])
    return sorted(k for k, splits in seen.items() if len(splits) > 1)


def load_sources(report_root):
    paths = [ROOT / "data/metadata" / name for name in
             ("manifest.json", "segmented_source.json", "nonleaf_manifest.json")]
    leaves_all, masks, backgrounds_all = map(read_json, paths)
    excluded, evidence = excluded_background_groups(report_root)
    leaves = [r for r in leaves_all if r["split"] in SPLITS]
    masks = {r["original_path"]: r for r in masks["files"] if r["split"] in SPLITS}
    backgrounds = [r for r in backgrounds_all if background_eligible(r, excluded)]
    # Metadata-only audit includes held-out partition; its pixels are never read.
    checks = {"leaf_group_cross_split": split_collisions(leaves_all, "group_id"),
              "leaf_hash_cross_split": split_collisions(leaves_all, "sha256"),
              "leaf_pixel_hash_cross_split": split_collisions(leaves_all, "pixel_sha256"),
              "background_scene_cross_split": split_collisions(backgrounds_all, "group_id"),
              "background_source_hash_cross_split": split_collisions(backgrounds_all, "source_sha256"),
              "background_crop_pixel_hash_cross_split": split_collisions(backgrounds_all, "pixel_sha256")}
    if any(checks.values()):
        raise ValueError(f"Cross-split source contamination: {checks}")
    for row in leaves:
        mask = masks[row["path"]]
        if mask["split"] != row["split"] or mask["group_id"] != row["group_id"]:
            raise ValueError("Mask source group/split changed")
        row["mask_path"] = mask["segmented_path"]
        row["mask_sha256"] = mask["sha256"]
    return leaves, backgrounds, {"input_manifests": [{"path": relative(p), "sha256": digest(p.read_bytes())} for p in paths],
        "qa_exclusion_evidence": evidence, "excluded_background_groups": sorted(excluded),
        "additional_background_visual_exclusions": VISUAL_BACKGROUND_EXCLUSIONS,
        "metadata_cross_split_checks": checks, "test_pixels_read": False,
        "background_split_policy": "Project grouped split, not the upstream COCO archive split; every source scene remains in one project split"}


@lru_cache(maxsize=2048)
def cached_leaf(path, mask_path):
    with Image.open(ROOT / path) as im:
        rgb = im.convert("RGB")
    with Image.open(ROOT / mask_path) as im:
        leaf, _ = foreground(im.convert("RGB"), rgb)
    return leaf


def verify_sources(leaves, backgrounds):
    source_audit = []
    for row in leaves + backgrounds:
        path = ROOT / row["path"]
        if digest(path.read_bytes()) != row["sha256"]:
            raise ValueError(f"Source file hash changed: {path}")
        with Image.open(path) as im:
            rgb = im.convert("RGB")
        decoded_hash = digest(str(rgb.size).encode() + rgb.tobytes())
        if "mask_path" in row and decoded_hash != row["pixel_sha256"]:
            raise ValueError(f"Source RGB hash changed: {path}")
        audit = {"path": row["path"], "sha256": row["sha256"], "split": row["split"], "group_id": row["group_id"]}
        if "mask_path" in row:
            mask_path = ROOT / row["mask_path"]
            if digest(mask_path.read_bytes()) != row["mask_sha256"]:
                raise ValueError(f"Source mask hash changed: {mask_path}")
            with Image.open(mask_path) as im:
                rgba, fraction = foreground(im.convert("RGB"), rgb)
            audit.update(mask_path=row["mask_path"], mask_sha256=row["mask_sha256"],
                         alpha_fraction=fraction, alpha_bbox_size=list(rgba.size))
        else:
            # Existing COCO crop pixel hashes describe the RGB before JPEG
            # encoding. Verify them against the immutable source scene crop,
            # while separately freezing the decoded JPEG actually composed.
            scene = cached_scene(row["source_path"], row["source_sha256"], row["source_pixel_sha256"])
            crop = scene.crop(row["crop_xyxy"])
            if digest(str(crop.size).encode() + crop.tobytes()) != row["pixel_sha256"]:
                raise ValueError(f"COCO pre-encoding crop hash changed: {path}")
            row["decoded_crop_pixel_sha256"] = decoded_hash
            audit.update(decoded_crop_pixel_sha256=decoded_hash,
                         upstream_pixel_hash_convention="sha256(str(size) + pre-JPEG RGB)",
                         source_path=row["source_path"], source_sha256=row["source_sha256"])
        source_audit.append(audit)
    return source_audit


@lru_cache(maxsize=128)
def cached_scene(path, expected_hash, expected_pixels):
    source = ROOT / path
    if digest(source.read_bytes()) != expected_hash:
        raise ValueError(f"COCO source hash changed: {path}")
    with Image.open(source) as im:
        rgb = im.convert("RGB")
    if digest(str(rgb.size).encode() + rgb.tobytes()) != expected_pixels:
        raise ValueError(f"COCO source RGB changed: {path}")
    return rgb


def alpha_blur(rgba, radius, motion):
    """Blur premultiplied RGB and alpha together, avoiding a grey/black halo."""
    values = np.asarray(rgba, dtype=np.float32)
    a = values[:, :, 3:4] / 255
    values[:, :, :3] *= a
    channels = []
    for channel in range(4):
        im = Image.fromarray(np.uint8(np.clip(values[:, :, channel], 0, 255)))
        if radius:
            im = im.filter(ImageFilter.GaussianBlur(radius))
        if motion:
            kernel = [0.] * 25
            kernel[10:15] = [1.] * 5
            im = im.filter(ImageFilter.Kernel((5, 5), kernel, scale=5))
        channels.append(np.asarray(im, dtype=np.float32))
    result = np.stack(channels, axis=2)
    result[:, :, :3] = np.divide(result[:, :, :3] * 255, result[:, :, 3:4],
                                  out=np.zeros_like(result[:, :, :3]), where=result[:, :, 3:4] > 0)
    return Image.fromarray(np.uint8(np.clip(result, 0, 255)), "RGBA")


def transform_leaf(source, rng):
    leaf = source.copy()
    flip = rng.random() < .5
    if flip:
        leaf = ImageOps.mirror(leaf)
    rgb = leaf.convert("RGB")
    settings = {"brightness": rng.uniform(.86, 1.14), "contrast": rng.uniform(.88, 1.12), "saturation": rng.uniform(.88, 1.12)}
    for name, enhancer in (("brightness", ImageEnhance.Brightness), ("contrast", ImageEnhance.Contrast), ("saturation", ImageEnhance.Color)):
        rgb = enhancer(rgb).enhance(settings[name])
    rgb.putalpha(leaf.getchannel("A"))
    aspect, angle, shear = rng.uniform(.82, 1.12), rng.uniform(-35, 35), rng.uniform(-.12, .12)
    leaf = rgb.resize((max(1, round(rgb.width * aspect)), rgb.height), Image.Resampling.BICUBIC)
    leaf = leaf.rotate(angle, Image.Resampling.BICUBIC, expand=True)
    shift = max(0, -shear * leaf.height)
    leaf = leaf.transform((leaf.width + math.ceil(abs(shear) * leaf.height), leaf.height),
                          Image.Transform.AFFINE, (1, -shear, -shift, 0, 1, 0), Image.Resampling.BICUBIC)
    leaf = leaf.crop(leaf.getchannel("A").getbbox())
    target = rng.randint(64, 230)
    scale = target / max(leaf.size)
    leaf = leaf.resize((max(1, round(leaf.width * scale)), max(1, round(leaf.height * scale))), Image.Resampling.LANCZOS)
    # Padding lets actual motion/defocus support expand without being cut off.
    padded = Image.new("RGBA", (leaf.width + 8, leaf.height + 8))
    padded.alpha_composite(leaf, (4, 4))
    blur, motion = rng.choice((0., 0., .35, .65)), rng.random() < .18
    leaf = alpha_blur(padded, blur, motion)
    leaf = leaf.crop(leaf.getchannel("A").getbbox())
    # Define scale by the final, transformed alpha support, not source rectangle.
    scale = target / max(leaf.size)
    leaf = leaf.resize((max(1, round(leaf.width * scale)), max(1, round(leaf.height * scale))), Image.Resampling.LANCZOS)
    return leaf, dict(settings, horizontal_flip=flip, aspect=aspect, angle_deg=angle, shear=shear,
                     gaussian_radius=blur, motion_kernel=5 if motion else 0, target_long_side=target)


def clipped_mask(leaf, position, size=SIZE):
    alpha = Image.new("L", size)
    alpha.paste(leaf.getchannel("A"), position)
    return np.asarray(alpha) > 0


def mask_bbox(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Empty transformed alpha")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def overlap_fraction(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return intersection / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


def valid_overlap(masks, boxes, candidate, box, limit=.20):
    if any(overlap_fraction(old, box) > limit for old in boxes):
        return False
    all_masks = masks + [candidate]
    for i, current in enumerate(all_masks):
        others = np.zeros_like(current)
        for j, other in enumerate(all_masks):
            if i != j:
                others |= other
        if (current & others).sum() / current.sum() > limit:
            return False
    return True


def place_leaf(leaf, rng, masks, boxes, truncated):
    w, h = SIZE
    for _ in range(120):
        x = rng.randint(4, w - leaf.width - 4)
        y = rng.randint(4, h - leaf.height - 4)
        if truncated:
            side = rng.randrange(4)
            amount = rng.uniform(.08, .30)
            if side == 0: x = -max(1, round(leaf.width * amount))
            if side == 1: x = w - leaf.width + max(1, round(leaf.width * amount))
            if side == 2: y = -max(1, round(leaf.height * amount))
            if side == 3: y = h - leaf.height + max(1, round(leaf.height * amount))
        mask = clipped_mask(leaf, (x, y))
        retained = mask.sum() / max(1, (np.asarray(leaf.getchannel("A")) > 0).sum())
        if retained < .65:
            continue
        box = mask_bbox(mask)
        if valid_overlap(masks, boxes, mask, box):
            return (x, y), mask, box, float(retained)
    raise ValueError("Could not place requested leaves within the overlap limit")


def yolo_line(box, size=SIZE):
    x1, y1, x2, y2 = box
    w, h = size
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("BBox lies outside the image")
    return "0 " + " ".join(f"{v:.9f}" for v in ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))


class SourceCycle:
    def __init__(self, rows, seed, key):
        self.rows = defaultdict(list)
        for row in sorted(rows, key=lambda r: r["path"]):
            self.rows[row[key]].append(row)
        self.keys = sorted(self.rows)
        self.cursor = Counter()
        rng = random.Random(seed)
        rng.shuffle(self.keys)
        for values in self.rows.values():
            rng.shuffle(values)

    def next(self, key):
        value = self.rows[key][self.cursor[key] % len(self.rows[key])]
        self.cursor[key] += 1
        return value


def compose_objects(background, sources, rng):
    planned = []
    for source in sources:
        leaf, pose = transform_leaf(cached_leaf(source["path"], source["mask_path"]), rng)
        planned.append((source, leaf, pose, rng.random() < .18))
    # Retry every position together: two unlucky central placements must not
    # force the third leaf to disappear or acquire an ambiguous ground truth.
    for attempt in range(25):
        canvas = background.copy()
        masks, boxes, instances = [], [], []
        try:
            for channel, (source, leaf, pose, truncated) in enumerate(planned):
                position, mask, box, retained = place_leaf(leaf, rng, masks, boxes, truncated)
                canvas.alpha_composite(leaf, position)
                masks.append(mask)
                boxes.append(box)
                instances.append({"class_id": 0, "name": "leaf", "source": leaf_provenance(source),
                                  "pose": pose, "position_xy": list(position), "transformed_size": list(leaf.size),
                                  "bbox_xyxy": box, "mask_channel": channel, "alpha_support_pixels": int(mask.sum()),
                                  "alpha_support_sha256": digest(mask.tobytes()), "retained_fraction": retained,
                                  "requested_truncated": truncated, "truncated": retained < 1})
            return canvas, masks, boxes, instances, attempt
        except ValueError as error:
            if "Could not place" not in str(error):
                raise
    raise ValueError("Could not arrange complete frame without ambiguous overlap")


def leaf_provenance(row):
    return dict({k: row[k] for k in ("path", "sha256", "pixel_sha256", "group_id", "split", "label", "mask_path", "mask_sha256")},
                **PLANT_SOURCE)


def background_provenance(row):
    return {k: row[k] for k in ("path", "sha256", "pixel_sha256", "decoded_crop_pixel_sha256", "group_id", "split", "source_scene_id", "source_path", "source_sha256", "source_pixel_sha256", "source_split", "source", "source_image_url", "flickr_source_url", "license", "crop_xyxy", "region_purity", "vegetation_fraction", "unknown_fraction", "audit_status", "category") if k in row}


def generate_split(split, count, leaves, backgrounds, out, seed):
    if count % 15:
        raise ValueError("Frame count must be divisible by 15 for exact 20% negatives and 1/2/3 balance")
    leaf_cycle = SourceCycle([r for r in leaves if r["split"] == split], seed, "label")
    bg_cycle = SourceCycle([r for r in backgrounds if r["split"] == split], seed + 1, "group_id")
    classes = sorted(leaf_cycle.rows)
    rows, positive_index = [], 0
    for frame_index in range(count):
        rng = random.Random(int.from_bytes(hashlib.sha256(f"{seed}:{split}:{frame_index}".encode()).digest()[:8], "little"))
        bg = bg_cycle.next(bg_cycle.keys[frame_index % len(bg_cycle.keys)])
        with Image.open(ROOT / bg["path"]) as image:
            canvas = ImageOps.fit(image.convert("RGB"), SIZE, Image.Resampling.BICUBIC,
                                  centering=(rng.random(), rng.random()))
        brightness = rng.uniform(.90, 1.10)
        canvas = ImageEnhance.Brightness(canvas).enhance(brightness).convert("RGBA")
        negative = frame_index % 5 == 0
        n = 0 if negative else 1 + positive_index % 3
        if not negative:
            positive_index += 1
        # Rotate the singleton class every block; the doubleton is its
        # complement. This balances all 3 classes exactly in each 1/2/3 block,
        # while preventing source class from predicting object count.
        block = (positive_index - 1) // 3
        start = (block + (1 if n == 2 else 0)) % 3
        sources = []
        for channel in range(n):
            sources.append(leaf_cycle.next(classes[(start + channel) % 3]))
        canvas, masks, boxes, instances, retries = compose_objects(canvas, sources, rng)
        for i, instance in enumerate(instances):
            later = np.zeros((SIZE[1], SIZE[0]), bool)
            for mask in masks[i + 1:]:
                later |= mask
            instance["occluded_fraction"] = float((masks[i] & later).sum() / masks[i].sum())
        stem = f"{split}_{frame_index:05d}"
        image_path, label_path, mask_path = (out / kind / split / (stem + suffix) for kind, suffix in
                                             (("images", ".jpg"), ("labels", ".txt"), ("masks", ".png")))
        canvas.convert("RGB").save(image_path, quality=94, subsampling=0)
        label_path.write_text("\n".join(yolo_line(box) for box in boxes) + ("\n" if boxes else ""), encoding="utf-8")
        packed = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
        for channel, mask in enumerate(masks):
            packed[:, :, channel] = mask.astype(np.uint8) * 255
        Image.fromarray(packed).save(mask_path)
        rows.append({"image": image_path.relative_to(out).as_posix(), "image_sha256": digest(image_path.read_bytes()),
                     "label": label_path.relative_to(out).as_posix(), "label_sha256": digest(label_path.read_bytes()),
                     "masks": mask_path.relative_to(out).as_posix(), "mask_sha256": digest(mask_path.read_bytes()),
                     "split": split, "frame_index": frame_index, "width": SIZE[0], "height": SIZE[1],
                     "negative": negative, "background": background_provenance(bg),
                     "background_brightness": brightness, "layout_retries": retries, "instances": instances})
        if frame_index % 300 == 0:
            print(f"{split}: {frame_index}/{count}", flush=True)
    return rows


def repair_duplicate_negatives(rows, backgrounds, out, report, seed):
    """Replace degenerate duplicate negatives using reviewed textured backgrounds.

    Saturated all-white sky crops from unrelated scenes may become identical.
    Keep their original files/records as evidence, never alter positive frames,
    and never add imperceptible noise merely to make hashes different.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row["image_sha256"]].append(row)
    duplicates = [r for group in groups.values() if len(group) > 1 for r in group]
    if not duplicates:
        return {"repaired_frames": [], "count": 0}
    if any(not r["negative"] or r["instances"] for r in duplicates):
        raise ValueError("Duplicate positive images require a fresh generation audit")
    evidence = report / "pre_duplicate_repair"
    evidence.mkdir(exist_ok=True)
    write_json(evidence / "original_frames.json", duplicates)
    unique = {r["image_sha256"] for r in rows if r not in duplicates}
    textured = defaultdict(list)
    for bg in backgrounds:
        with Image.open(ROOT / bg["path"]) as im:
            rgb = im.convert("RGB")
        if float(np.asarray(rgb, dtype=np.float32).std(axis=(0, 1)).mean()) >= 6:
            textured[bg["split"]].append(bg)
    repaired = []
    for row in duplicates:
        original = json.loads(json.dumps(row))
        old_path = out / row["image"]
        (evidence / old_path.name).write_bytes(old_path.read_bytes())
        for attempt in range(100):
            derived_seed = int.from_bytes(hashlib.sha256(f"{seed}:negative_duplicate:{row['split']}:{row['frame_index']}:{attempt}".encode()).digest()[:8], "little")
            rng = random.Random(derived_seed)
            bg = rng.choice(textured[row["split"]])
            if digest((ROOT / bg["path"]).read_bytes()) != bg["sha256"]:
                raise ValueError("Background source changed during negative repair")
            with Image.open(ROOT / bg["path"]) as im:
                rgb = im.convert("RGB")
            bg["decoded_crop_pixel_sha256"] = digest(str(rgb.size).encode() + rgb.tobytes())
            center = (rng.random(), rng.random())
            brightness = rng.uniform(.90, 1.10)
            canvas = ImageEnhance.Brightness(ImageOps.fit(rgb, SIZE, Image.Resampling.BICUBIC, centering=center)).enhance(brightness)
            texture = float(np.asarray(canvas, dtype=np.float32).std(axis=(0, 1)).mean())
            if texture < 6:
                continue
            from io import BytesIO
            buffer = BytesIO()
            canvas.save(buffer, format="JPEG", quality=94, subsampling=0)
            encoded = buffer.getvalue()
            checksum = digest(encoded)
            if checksum in unique:
                continue
            old_path.write_bytes(encoded)
            row.update(image_sha256=checksum, background=background_provenance(bg), background_brightness=brightness,
                       background_centering_xy=list(center), negative_repair={"reason": "duplicate_flat_negative",
                           "original_image_sha256": original["image_sha256"], "attempt": attempt,
                           "derived_seed": derived_seed, "spatial_rgb_std": texture})
            unique.add(checksum)
            repaired.append({"image": row["image"], "original_image_sha256": original["image_sha256"],
                             "image_sha256": checksum, "original_background_group": original["background"]["group_id"],
                             "background_group": bg["group_id"], "split": row["split"], "spatial_rgb_std": texture})
            break
        else:
            raise ValueError("No unique textured background could replace a flat negative")
    return {"count": len(repaired), "repaired_frames": repaired,
            "policy": "No positive pixels/GT changed; each replacement uses a reviewed textured background of the same project split"}


def audit_generated(rows, out):
    checks = Counter()
    errors = []
    sources = []
    for row in rows:
        for name, hash_key in (("image", "image_sha256"), ("label", "label_sha256"), ("masks", "mask_sha256")):
            if digest((out / row[name]).read_bytes()) != row[hash_key]:
                errors.append(f"hash:{row[name]}")
        with Image.open(out / row["image"]) as image:
            if image.size != SIZE or image.mode != "RGB": errors.append(f"size/mode:{row['image']}")
        with Image.open(out / row["masks"]) as image:
            packed = np.asarray(image)
        lines = (out / row["label"]).read_text().splitlines()
        if len(lines) != len(row["instances"]): errors.append(f"label_count:{row['image']}")
        if row["negative"] != (len(lines) == 0): errors.append(f"negative:{row['image']}")
        if row["negative"] and packed.any(): errors.append(f"negative_mask:{row['image']}")
        masks, boxes = [], []
        for instance, line in zip(row["instances"], lines):
            mask = packed[:, :, instance["mask_channel"]] > 0
            box = mask_bbox(mask)
            if box != instance["bbox_xyxy"] or line != yolo_line(box): errors.append(f"bbox_alpha:{row['image']}")
            if digest(mask.tobytes()) != instance["alpha_support_sha256"]: errors.append(f"alpha_hash:{row['image']}")
            if not valid_overlap(masks, boxes, mask, box): errors.append(f"overlap:{row['image']}")
            if instance["retained_fraction"] < .65 or instance["occluded_fraction"] > .20: errors.append(f"visibility:{row['image']}")
            if instance["source"]["split"] != row["split"]: errors.append(f"leaf_split:{row['image']}")
            sources.append(dict(instance["source"], kind="leaf"))
            masks.append(mask)
            boxes.append(box)
            checks["instances"] += 1
        if row["background"]["split"] != row["split"]: errors.append(f"background_split:{row['image']}")
        sources.append(dict(row["background"], kind="background"))
        checks["frames"] += 1
        checks["negative_frames"] += row["negative"]
    collision_checks = {key: split_collisions(sources, key) for key in ("group_id", "sha256", "pixel_sha256")}
    if any(collision_checks.values()): errors.append("generated_source_cross_split")
    duplicate_images = [key for key, count in Counter(r["image_sha256"] for r in rows).items() if count > 1]
    if duplicate_images: errors.append("duplicate_generated_images")
    return {"passed": not errors, "errors": errors, "counts": dict(checks),
            "generated_source_cross_split": collision_checks, "duplicate_generated_image_hashes": duplicate_images}


def write_contacts(items, target, prefix, thumbnail, per_page=30):
    sheets = []
    for page in range(math.ceil(len(items) / per_page)):
        subset = items[page * per_page:(page + 1) * per_page]
        width, height = thumbnail
        sheet = Image.new("RGB", (width * 5, (height + 32) * 6), (246, 246, 246))
        draw = ImageDraw.Draw(sheet)
        for i, (path, caption, boxes) in enumerate(subset):
            x, y = i % 5 * width, i // 5 * (height + 32)
            with Image.open(path) as im:
                im = ImageOps.contain(im.convert("RGB"), thumbnail)
            sheet.paste(im, (x + (width - im.width) // 2, y + (height - im.height) // 2))
            for box in boxes:
                draw.rectangle((x + box[0] * width / 640, y + box[1] * height / 360,
                                x + box[2] * width / 640, y + box[3] * height / 360), outline="red", width=2)
            draw.text((x + 3, y + height + 2), caption, fill="black")
        path = target / f"{prefix}_{page:02d}.jpg"
        sheet.save(path, quality=95)
        sheets.append({"path": relative(path), "sha256": digest(path.read_bytes()), "items": len(subset)})
    return sheets


def make_contacts(rows, backgrounds, out, report, seed):
    # Exactly 10% in every split x leaf-count stratum, not only easy positives.
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["split"], len(row["instances"]))].append(row)
    selected = []
    rng = random.Random(seed)
    for key, values in sorted(buckets.items()):
        selected.extend(rng.sample(values, math.ceil(len(values) * .10)))
    selected.sort(key=lambda r: (r["split"], r["frame_index"]))
    items = [(out / r["image"], f"{Path(r['image']).stem} n={len(r['instances'])} " +
              ("NEGATIVE" if r["negative"] else "edge" if any(i["truncated"] for i in r["instances"]) else ""),
              [i["bbox_xyxy"] for i in r["instances"]]) for r in selected]
    scene_sheets = write_contacts(items, report, "scenes", (256, 144))
    bg_items = [(ROOT / r["path"], f"{r['split']} {r['group_id']} {r['category'][:18]}", []) for r in backgrounds]
    bg_sheets = write_contacts(bg_items, report, "backgrounds", (192, 128))
    return {"sample_rule": "10% per split x object-count (0/1/2/3) with fixed seed; all eligible background crops also shown",
            "selected_images": [r["image"] for r in selected], "selected_count": len(selected),
            "scene_sheets": scene_sheets, "all_background_sheets": bg_sheets,
            "sample_distribution": {f"{s}_n{n}": sum(r["split"] == s and len(r["instances"]) == n for r in selected) for s, n in sorted(buckets)}}


def summarize(rows, leaves, backgrounds):
    summary = {}
    for split in SPLITS:
        frames = [r for r in rows if r["split"] == split]
        instances = [i for r in frames for i in r["instances"]]
        summary[split] = {"frames": len(frames), "negative_frames": sum(r["negative"] for r in frames),
            "object_count_distribution": dict(Counter(len(r["instances"]) for r in frames)),
            "instances": len(instances), "source_class_instances": dict(Counter(i["source"]["label"] for i in instances)),
            "available_leaf_photos": sum(r["split"] == split for r in leaves),
            "used_leaf_photos": len(set(i["source"]["path"] for i in instances)),
            "available_leaf_groups": len(set(r["group_id"] for r in leaves if r["split"] == split)),
            "used_leaf_groups": len(set(i["source"]["group_id"] for i in instances)),
            "background_crops": sum(r["split"] == split for r in backgrounds),
            "background_scenes": len(set(r["group_id"] for r in backgrounds if r["split"] == split)),
            "truncated_instances": sum(i["truncated"] for i in instances),
            "motion_instances": sum(bool(i["pose"]["motion_kernel"]) for i in instances),
            "maximum_occluded_fraction": max(i["occluded_fraction"] for i in instances)}
    return summary


def review_visual(out, report, note):
    audit = read_json(report / "audit.json")
    contacts = read_json(report / "contacts.json")
    if not audit["passed"]:
        raise ValueError("Numerical audit failed; visual review cannot authorize training")
    for sheet in contacts["scene_sheets"] + contacts["all_background_sheets"]:
        if digest((ROOT / sheet["path"]).read_bytes()) != sheet["sha256"]:
            raise ValueError("Contact sheet changed since generation")
    manifest_hash = digest((out / "manifest.json").read_bytes())
    if audit["manifest_sha256"] != manifest_hash:
        raise ValueError("Manifest changed since numerical audit")
    # A second disk audit prevents changed images/labels after contact rendering.
    current = audit_generated(read_json(out / "manifest.json"), out)
    if not current["passed"]:
        raise ValueError(current["errors"])
    review = {"reviewer": "assistant visual inspection; not an agricultural expert claim",
              "decision": "approved_for_local_detector_candidate_training", "note": note,
              "all_contact_sheets_reviewed": True, "contacts_sha256": digest((report / "contacts.json").read_bytes()),
              "manifest_sha256": manifest_hash, "can_train": True,
              "dataset_yaml_sha256": digest((out / "dataset.yaml").read_bytes()),
              "audit_sha256": digest((report / "audit.json").read_bytes()),
              "protocol_sha256": digest((report / "protocol.json").read_bytes()),
              "evaluation_scope": "Synthetic grouped validation reused for development; no independent field accuracy claim"}
    write_json(report / "visual_review.json", review)
    write_json(out / "training_readiness.json", review)
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--train", type=int, default=1800)
    parser.add_argument("--val", type=int, default=300)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--review-visual", help="Only after inspecting every contact sheet: concrete review note")
    args = parser.parse_args()
    out, report = args.out.resolve(), args.report.resolve()
    if args.review_visual:
        print(json.dumps(review_visual(out, report, args.review_visual), indent=2))
        return
    if out.exists() and any(p.is_file() for p in out.rglob("*")):
        raise FileExistsError(f"Preserve existing generated dataset: {out}")
    if report.exists() and any(report.iterdir()):
        raise FileExistsError(f"Preserve existing report: {report}")
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    for kind in ("images", "labels", "masks"):
        for split in SPLITS:
            (out / kind / split).mkdir(parents=True, exist_ok=True)
    leaves, backgrounds, protocol = load_sources(ROOT / "reports")
    print(f"Checking {len(leaves)} leaf photos and {len(backgrounds)} background crops", flush=True)
    write_json(report / "source_audit.json", verify_sources(leaves, backgrounds))
    protocol.update(seed=args.seed, canvas=list(SIZE), frames={"train": args.train, "val": args.val},
        negative_fraction=.20, instance_count_range=[1, 3], leaf_long_side_range=[64, 230],
        class_schedule="rotate singleton class per 1/2/3 block; doubleton uses its complement; each source class exactly balanced per block",
        layout_policy="retry entire layout up to25 times, never drop a requested leaf or relax the overlap bound",
        max_pairwise_bbox_overlap_over_smaller=.20, max_cumulative_alpha_overlap=.20,
        min_visible_alpha_fraction=.65, alpha_support_rule="all nonzero alpha after all transforms/blur and canvas clipping",
        labels="one class: 0 leaf; normalized cx cy w h; empty .txt for negatives",
        format_reference="https://docs.ultralytics.com/datasets/detect/",
        plant_provenance=PLANT_SOURCE, leaf_rgb_policy="original RGB; segmented JPEG supplies geometry only",
        no_person_veto=True, no_model_or_runtime_changes=True, training_started=False,
        generator_sha256=digest(Path(__file__).read_bytes()))
    write_json(report / "protocol.json", protocol)
    rows = []
    for split, count in (("train", args.train), ("val", args.val)):
        rows.extend(generate_split(split, count, leaves, backgrounds, out, args.seed))
    write_json(report / "duplicate_negative_repairs.json", repair_duplicate_negatives(rows, backgrounds, out, report, args.seed))
    write_json(out / "manifest.json", rows)
    (out / "dataset.yaml").write_text(f"# Synthetic potato leaves; grouped source partitions; local research\npath: {json.dumps(out.as_posix())}\ntrain: images/train\nval: images/val\nnames:\n  0: leaf\n", encoding="utf-8")
    audit = audit_generated(rows, out)
    audit["manifest_sha256"] = digest((out / "manifest.json").read_bytes())
    write_json(report / "audit.json", audit)
    write_json(report / "summary.json", summarize(rows, leaves, backgrounds))
    write_json(report / "contacts.json", make_contacts(rows, backgrounds, out, report, args.seed))
    write_json(out / "training_readiness.json", {"can_train": False, "reason": "Visual review pending", "numerical_audit_passed": audit["passed"]})
    (out / "README.md").write_text(
        "# Potato leaf detection candidate data\n\n"
        "Synthetic 640x360 compositions; one detection class (`0: leaf`). "
        "RGB comes from original PlantVillage potato photos and alpha geometry from their matching segmented JPEGs. "
        "Dark enclosed disease tissue stays opaque. Only already-reviewed non-leaf COCO crops from the same project split are backgrounds.\n\n"
        "`manifest.json` retains source groups, hashes, image attribution/license fields, transforms and clipped boxes. "
        "`masks/` stores binary transformed alpha support in RGB channels (one channel per instance), before limited occlusion. "
        "Negative images have empty label files and black mask images. "
        "`training_readiness.json` must have `can_train: true` and matching hashes before starting a candidate training run.\n\n"
        "The grouped validation set is reused development data. Synthetic validation accuracy does not establish field or moving-rover accuracy. "
        "No held-out test pixels, user photos, cameras or models are used by this generator.\n\n"
        "PlantVillage: pinned repository commit and its open-access statement are retained; the pinned README gives no explicit image license. "
        "COCO/Flickr licenses are individual and include NonCommercial and ShareAlike conditions. "
        "This local research collection is not a blanket CC-BY image release. Attribution is in each frame's sources.\n",
        encoding="utf-8")
    print(json.dumps({"audit_passed": audit["passed"], "errors": audit["errors"], "report": str(report), "can_train": False}, indent=2))


if __name__ == "__main__":
    main()
