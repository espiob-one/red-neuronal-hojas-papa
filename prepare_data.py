"""Download only original potato RGB images and make leaf-grouped data splits."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
META = DATA / "metadata"
REPO = "spMohanty/PlantVillage-Dataset"
SOURCE_COMMIT = "7f7ecc7e1eaca78107e3affe7cb5abd9427e139a"
CLASSES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
SEED = 42


def fetch(url):
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": "rover-training-dataset-setup"}), timeout=45) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def fetch_json(url):
    return json.loads(fetch(url))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def discover():
    source_path = META / "source.json"
    if source_path.exists():
        return json.loads(source_path.read_text(encoding="utf-8"))
    api = f"https://api.github.com/repos/{REPO}"
    commit = SOURCE_COMMIT
    dirs = fetch_json(f"{api}/contents/raw/color?ref={commit}")
    files = []
    for label in CLASSES:
        directory = next(item for item in dirs if item["name"] == label and item["type"] == "dir")
        tree = fetch_json(f"{api}/git/trees/{directory['sha']}")
        if tree.get("truncated"):
            raise RuntimeError(f"Incomplete file listing: {label}")
        for item in tree["tree"]:
            if item["type"] == "blob" and Path(item["path"]).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                if Path(item["path"]).name != item["path"]:
                    raise RuntimeError("Unexpected nested image path")
                files.append({"label": label, "filename": item["path"], "git_sha": item["sha"], "bytes": item["size"]})
    source = {"repository": f"https://github.com/{REPO}", "commit": commit, "seed": SEED, "files": files}
    save_json(source_path, source)
    return source


def download(item, commit):
    path = DATA / "raw" / item["label"] / item["filename"]
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.read_bytes() if path.exists() else fetch(
        f"https://raw.githubusercontent.com/{REPO}/{commit}/raw/color/{quote(item['label'])}/{quote(item['filename'])}"
    )
    git_sha = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if git_sha != item["git_sha"]:
        raise RuntimeError(f"Source integrity mismatch: {path}")
    if not path.exists():
        path.write_bytes(raw)
    return path


def leaf_id(filename, label, leaf_map):
    identifier = filename.replace("_final_masked", "").split("___")[-1]
    identifier = identifier.split("copy")[0]
    identifier = identifier.replace(".jpg", "").replace(".JPG", "").replace(".png", "").replace(".PNG", "").strip().lower()
    choices = leaf_map.get(identifier, [])
    matching = [choice for choice in choices if choice.startswith(label + ":::")]
    return matching[0] if len(matching) == 1 else None


def main():
    META.mkdir(parents=True, exist_ok=True)
    source = discover()
    print("Source commit:", source["commit"], flush=True)
    print("Images:", dict(Counter(item["label"] for item in source["files"])), flush=True)
    map_path = META / "leaf-map.json"
    if not map_path.exists():
        map_path.write_bytes(fetch(f"https://raw.githubusercontent.com/{REPO}/{source['commit']}/leaf-map.json"))
    leaf_map = json.loads(map_path.read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(download, item, source["commit"]) for item in source["files"]]
        for count, job in enumerate(as_completed(jobs), 1):
            job.result()
            if count % 100 == 0 or count == len(jobs):
                print(f"Downloaded and verified: {count}/{len(jobs)}", flush=True)

    rows, excluded = [], []
    seen_pixels = {}
    parents = {}

    def group(value):
        parents.setdefault(value, value)
        if parents[value] != value:
            parents[value] = group(parents[value])
        return parents[value]

    for item in sorted(source["files"], key=lambda row: (row["label"], row["filename"])):
        path = DATA / "raw" / item["label"] / item["filename"]
        leaf = leaf_id(item["filename"], item["label"], leaf_map)
        if leaf is None:
            excluded.append({**item, "reason": "missing_or_ambiguous_leaf_group"})
            continue
        with Image.open(path) as image:
            image.load()
            rgb = image.convert("RGB")
            pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            width, height = rgb.size
        group(leaf)
        if pixels in seen_pixels:
            other = seen_pixels[pixels]
            if other["label"] != item["label"]:
                raise RuntimeError("Identical image pixels have conflicting labels")
            parents[group(leaf)] = group(other["leaf_id"])
            excluded.append({**item, "reason": "identical_pixels", "same_as": other["path"]})
            continue
        row = {
            "path": path.relative_to(ROOT).as_posix(), "label": item["label"],
            "class_index": CLASSES.index(item["label"]), "leaf_id": leaf,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pixel_sha256": pixels,
            "width": width, "height": height, "bytes": path.stat().st_size,
        }
        seen_pixels[pixels] = row
        rows.append(row)

    for row in rows:
        row["group_id"] = group(row["leaf_id"])
    rng = random.Random(SEED)
    assignments = {}
    for label in CLASSES:
        groups = sorted({row["group_id"] for row in rows if row["label"] == label})
        if len(groups) < 10:
            raise RuntimeError(f"Too few independent leaves: {label}")
        rng.shuffle(groups)
        n_test = max(1, round(len(groups) * .15))
        n_val = max(1, round(len(groups) * .15))
        for index, identity in enumerate(groups):
            assignments[identity] = "test" if index < n_test else "val" if index < n_test + n_val else "train"
    for row in rows:
        row["split"] = assignments[row["group_id"]]
    for column in ("group_id", "sha256", "pixel_sha256"):
        split_sets = {split: {row[column] for row in rows if row["split"] == split} for split in ("train", "val", "test")}
        if any(split_sets[a] & split_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise RuntimeError(f"Leakage detected for {column}")

    profile = {
        "source_commit": source["commit"], "seed": SEED, "grain": "one original RGB image per row",
        "source_images": len(source["files"]), "usable_images": len(rows),
        "excluded_images": len(excluded), "exclusion_reasons": dict(Counter(row["reason"] for row in excluded)),
        "bytes": sum(row["bytes"] for row in rows),
        "dimensions": dict(Counter(f"{row['width']}x{row['height']}" for row in rows)),
        "classes": {label: {"images": sum(row["label"] == label for row in rows),
            "leaf_groups": len({row["group_id"] for row in rows if row["label"] == label}),
            "splits": dict(Counter(row["split"] for row in rows if row["label"] == label))} for label in CLASSES},
        "missing_manifest_values": sum(value is None or value == "" for row in rows for value in row.values()),
        "duplicate_paths": len(rows) - len({row["path"] for row in rows}),
        "split_policy": "70/15/15 by leaf groups within each class; exact pixel duplicates removed; originals preserved",
        "cross_split_group_or_hash_overlap": 0,
        "limitations": ["Controlled backgrounds, not field validation", "Few healthy leaves relative to diseased leaves", "Three labels only; no unknown/non-leaf class"],
    }
    save_json(META / "manifest.json", rows)
    save_json(META / "excluded.json", excluded)
    save_json(META / "profile.json", profile)
    save_json(META / "classes.json", CLASSES)

    sheet = Image.new("RGB", (5 * 170, 3 * 200), "white")
    draw = ImageDraw.Draw(sheet)
    for y, label in enumerate(CLASSES):
        samples = sorted((row for row in rows if row["label"] == label), key=lambda row: row["path"])
        for x, row in enumerate(samples[::max(1, len(samples) // 5)][:5]):
            with Image.open(ROOT / row["path"]) as image:
                image = image.convert("RGB")
                image.thumbnail((160, 160))
                sheet.paste(image, (x * 170, y * 200 + 25))
        draw.text((5, y * 200 + 5), label, fill="black")
    sheet.save(META / "sample_sheet.jpg")
    print(json.dumps(profile, indent=2), flush=True)


if __name__ == "__main__":
    main()
