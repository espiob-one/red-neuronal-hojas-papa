"""Isolated local YOLO toolchain; never installs into the existing venv.

All optional wheels go under tools/yolo-packages using --no-deps. Existing
Torch/TorchVision/NumPy distributions are protected and checked before/after.
The only model download is a digest-verified official Ultralytics GitHub asset.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from urllib.request import Request, urlopen

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "tools/yolo-packages"
WHEELS = ROOT / "tools/yolo-wheels"
SETTINGS = ROOT / "tools/settings"
WEIGHTS = ROOT / "tools/yolo-weights/yolov8n.pt"
REPORT = ROOT / "tools/yolo-toolchain.json"
PROTECTED = {"torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128", "numpy": "2.2.6"}
ULTRALYTICS_VERSION = "8.4.176"
ASSET_TAG = "v8.4.0"
ASSET_ID = 340060940
ASSET_SIZE = 6549796
ASSET_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"
ASSET_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt"


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def distributions(path=None):
    items = metadata.distributions() if path is None else metadata.distributions(path=path)
    return {canonicalize_name(item.metadata["Name"]): item for item in items}


def base_snapshot():
    available = distributions()
    result = {}
    for name, expected in PROTECTED.items():
        distribution = available[name]
        if distribution.version != expected:
            raise RuntimeError(f"Protected base version changed: {name}")
        meta_path = next(Path(distribution.locate_file(file)) for file in distribution.files if file.name == "METADATA" and file.parent.name.endswith(".dist-info"))
        result[name] = {"version": distribution.version, "metadata_path": str(meta_path), "metadata_sha256": digest(meta_path)}
    if TARGET.exists() and any(name in distributions([str(TARGET)]) for name in PROTECTED):
        raise RuntimeError("A target package shadows a protected base distribution")
    return result


def missing_dependencies():
    target = distributions([str(TARGET)])
    base = distributions()
    missing = {}
    if target.get("ultralytics") is None:
        missing["ultralytics"] = f"ultralytics=={ULTRALYTICS_VERSION}"
    elif target["ultralytics"].version != ULTRALYTICS_VERSION:
        raise RuntimeError("Unexpected isolated Ultralytics version")
    for distribution in target.values():
        for text in distribution.requires or []:
            requirement = Requirement(text)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            name = canonicalize_name(requirement.name)
            found = target.get(name, base.get(name))
            if found is not None and found.version in requirement.specifier:
                continue
            if name in PROTECTED:
                raise RuntimeError("A dependency conflicts with protected base versions")
            # Do not activate optional extras from any package.
            missing[name] = requirement.name + str(requirement.specifier)
    return sorted(missing.values())


def install_missing():
    TARGET.mkdir(parents=True, exist_ok=True)
    WHEELS.mkdir(parents=True, exist_ok=True)
    before = base_snapshot()
    for round_number in range(8):
        requirements = missing_dependencies()
        if not requirements:
            if base_snapshot() != before:
                raise RuntimeError("Protected base files changed")
            return
        print(json.dumps({"stage": "isolated_dependencies", "round": round_number+1, "requirements": requirements}), flush=True)
        command = [sys.executable, "-m", "pip", "--disable-pip-version-check", "download", "--no-deps", "--only-binary=:all:",
                   "--index-url", "https://pypi.org/simple", "--dest", str(WHEELS)] + requirements
        subprocess.run(command, check=True, timeout=300)
        selected = []
        for spec in requirements:
            requirement = Requirement(spec)
            candidates = []
            for path in WHEELS.glob("*.whl"):
                name, version, _, _ = parse_wheel_filename(path.name)
                if canonicalize_name(name) == canonicalize_name(requirement.name) and version in requirement.specifier:
                    candidates.append((version, path))
            if not candidates:
                raise RuntimeError("Downloaded wheel is missing")
            selected.append(max(candidates, key=lambda pair: pair[0])[1])
        for path in selected:
            name, _, _, _ = parse_wheel_filename(path.name)
            if canonicalize_name(name) in PROTECTED:
                raise RuntimeError("Refusing to install a protected package")
        subprocess.run([sys.executable, "-m", "pip", "--disable-pip-version-check", "install", "--no-deps", "--no-index",
                        "--upgrade", "--target", str(TARGET)] + [str(path) for path in selected], check=True, timeout=300)
        if base_snapshot() != before:
            raise RuntimeError("Protected base files changed")
    raise RuntimeError("Dependency resolution did not converge")


def download_official_weights():
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    api = "https://api.github.com/repos/ultralytics/assets/releases/tags/" + ASSET_TAG
    with urlopen(Request(api, headers={"User-Agent": "rover-yolo-isolated-setup", "Accept": "application/vnd.github+json"}), timeout=30) as response:
        release = json.load(response)
    asset = next(item for item in release["assets"] if item["id"] == ASSET_ID and item["name"] == "yolov8n.pt")
    if (asset.get("digest") != "sha256:" + ASSET_SHA256 or asset.get("size") != ASSET_SIZE or
            asset.get("browser_download_url") != ASSET_URL or release.get("tag_name") != ASSET_TAG):
        raise RuntimeError("Official asset provenance differs from the frozen digest")
    if not WEIGHTS.exists():
        temporary = WEIGHTS.with_suffix(".pending")
        try:
            with urlopen(Request(ASSET_URL, headers={"User-Agent": "rover-yolo-isolated-setup"}), timeout=60) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if temporary.stat().st_size != ASSET_SIZE or digest(temporary) != ASSET_SHA256:
                raise RuntimeError("Official weight digest mismatch")
            temporary.replace(WEIGHTS)
        finally:
            temporary.unlink(missing_ok=True)
    if WEIGHTS.stat().st_size != ASSET_SIZE or digest(WEIGHTS) != ASSET_SHA256:
        raise RuntimeError("Local official weight digest mismatch")
    return {"repository": "ultralytics/assets", "tag": ASSET_TAG, "asset_id": ASSET_ID,
            "url": ASSET_URL, "sha256": ASSET_SHA256, "bytes": ASSET_SIZE,
            "github_digest": asset["digest"], "path": str(WEIGHTS)}


def configure_environment():
    """Process-local paths/environment only; no permanent user settings."""
    SETTINGS.mkdir(parents=True, exist_ok=True)
    config_dir = SETTINGS/"Ultralytics"  # 8.4.176 appends this to YOLO_CONFIG_DIR.
    config_dir.mkdir(parents=True, exist_ok=True)
    settings = {"settings_version": "0.0.8", "datasets_dir": str(ROOT/"data"), "weights_dir": str(WEIGHTS.parent),
                "runs_dir": str(ROOT/"runs/potato_detector_candidates"), "uuid": hashlib.sha256(b"rover-local-only").hexdigest(),
                "sync": False, "api_key": "", "openai_api_key": "", "clearml": False, "comet": False,
                "dvc": False, "mlflow": False, "raytune": False, "tensorboard": False, "wandb": False,
                "vscode_msg": False, "openvino_msg": False}
    settings_path = config_dir/"settings.json"
    if not settings_path.exists() or json.loads(settings_path.read_text()) != settings:
        save(settings_path, settings)
    os.environ.update({"YOLO_CONFIG_DIR": str(SETTINGS), "YOLO_OFFLINE": "true", "YOLO_AUTOINSTALL": "false",
                       "MPLCONFIGDIR": str(SETTINGS/"matplotlib"), "MPLBACKEND": "Agg", "WANDB_MODE": "disabled",
                       "WANDB_DISABLED": "true", "COMET_MODE": "DISABLED", "ULTRALYTICS_API_KEY": "",
                       "ULTRALYTICS_HUB_API_KEY": "", "HF_HUB_OFFLINE": "1"})
    if str(TARGET) not in sys.path:
        sys.path.insert(0, str(TARGET))
    # check_det_dataset always checks Arial, even with plots=False. Provide an
    # existing local font so that the library never tries to download one.
    font = config_dir/"Arial.ttf"
    if not font.exists():
        candidates = [Path(os.environ.get("WINDIR", "C:/Windows"))/"Fonts/arial.ttf",
                      TARGET/"matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"]
        local = next((path for path in candidates if path.is_file()), None)
        if local is None:
            raise RuntimeError("No local font available for offline dataset checks")
        shutil.copyfile(local, font)
        save(config_dir/"font-source.json", {"path": str(local), "sha256": digest(local), "local_only": True})
    return settings


@contextmanager
def offline_network():
    """Deny network and DNS inside the current import/train/export process."""
    originals = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo
    blocked = []
    def deny(*args, **kwargs):
        blocked.append(True)
        raise OSError("YOLO training toolchain runs offline")
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.getaddrinfo = deny
    try:
        yield blocked
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = originals


def activate():
    before = base_snapshot()
    if missing_dependencies():
        raise RuntimeError("Run setup_yolo_tools.py --install before using YOLO")
    settings = configure_environment()
    return before, settings


def check_imports():
    before, settings = activate()
    with offline_network() as blocked:
        import cv2
        import matplotlib
        import polars
        import numpy
        import torch
        import torchvision
        from ultralytics import YOLO
        import ultralytics
        from ultralytics.utils import SETTINGS as live_settings, ONLINE, AUTOINSTALL
        from ultralytics.utils.events import events
        assert not ONLINE and not AUTOINSTALL and not events.enabled
        assert all(not live_settings[key] for key in ("sync", "clearml", "comet", "dvc", "mlflow", "raytune", "tensorboard", "wandb"))
        assert torch.__version__ == PROTECTED["torch"] and torchvision.__version__ == PROTECTED["torchvision"] and numpy.__version__ == PROTECTED["numpy"]
        assert not Path(torch.__file__).is_relative_to(TARGET) and not Path(numpy.__file__).is_relative_to(TARGET)
        versions = {"ultralytics": ultralytics.__version__, "cv2": cv2.__version__, "matplotlib": matplotlib.__version__, "polars": polars.__version__,
                    "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": numpy.__version__}
        frame = polars.DataFrame({"ready": [True]})
        assert frame.height == 1 and YOLO is not None
    if base_snapshot() != before:
        raise RuntimeError("Protected base distribution changed during import")
    return {"versions": versions, "settings": settings, "network_attempts_blocked": len(blocked), "protected_base": before,
            "scipy_required": False, "passed": True}


def record_toolchain(check, weight_info=None):
    packages = []
    for name, distribution in sorted(distributions([str(TARGET)]).items()):
        files = []
        for file in distribution.files or []:
            path = Path(distribution.locate_file(file))
            if path.is_file() and path.is_relative_to(TARGET) and path.suffix not in (".pyc",):
                files.append({"path": str(path.relative_to(TARGET)), "sha256": digest(path)})
        packages.append({"name": name, "version": distribution.version,
                         "installed_files_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
                         "file_count": len(files)})
    previous = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "target": str(TARGET), "check": check,
              "packages": packages, "wheels": [{"filename": path.name, "sha256": digest(path), "bytes": path.stat().st_size} for path in sorted(WHEELS.glob("*.whl"))],
              "weights": weight_info or previous.get("weights"), "setup_source_sha256": digest(__file__),
              "mutated_base": False, "trained": False}
    save(REPORT, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--download-weights", action="store_true")
    args = parser.parse_args()
    before = base_snapshot()
    if args.install:
        install_missing()
    weights = download_official_weights() if args.download_weights else None
    check = check_imports()
    report = record_toolchain(check, weights)
    if base_snapshot() != before:
        raise RuntimeError("Protected base versions changed")
    print(json.dumps({"passed": True, "trained": False, "report": str(REPORT), "versions": check["versions"],
                      "network_attempts_blocked": check["network_attempts_blocked"], "weights_ready": report.get("weights") is not None}), flush=True)


if __name__ == "__main__":
    main()
