"""
Download and organize the Kaggle dataset used by this repository.

Kaggle dataset:
    luisngeld/person-detection-uav-pascal-voc-uaq-msuav

Final structure generated under the repository root:
    DATASET_KAGGLE/
      PASCAL_VOC_UAQ_MSUAV/
        images/
        labels/
      EVALUATION/
        COCO_TEST/images
        COCO_TEST/labels
        IRINA/images
        IRINA/labels
        MANIPAL_UAV/images
        MANIPAL_UAV/labels
        NTUT/images
        NTUT/labels
        UAQ_MSUAV_TEST/images
        UAQ_MSUAV_TEST/labels
        VISDRONE/images
        VISDRONE/labels
      evaluation_videos/
        VIDEO_1/
        VIDEO_2/
"""

from __future__ import annotations

import os
import shutil
import ssl
from pathlib import Path

import kagglehub
import requests
import urllib3


KAGGLE_DATASET = "luisngeld/person-detection-uav-pascal-voc-uaq-msuav"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "DATASET_KAGGLE"

TRAINING_DATASET = "PASCAL_VOC_UAQ_MSUAV"
EVAL_DATASETS = (
    "COCO_TEST",
    "IRINA",
    "MANIPAL_UAV",
    "NTUT",
    "UAQ_MSUAV_TEST",
    "VISDRONE",
)
REQUIRED_VIDEO_DIRS = ("VIDEO_1", "VIDEO_2")


def configure_kaggle_environment() -> None:
    os.environ.setdefault("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))

    if os.environ.get("KAGGLE_INSECURE_SSL", "").strip() != "1":
        return

    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["PYTHONHTTPSVERIFY"] = "0"
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    original_merge_environment_settings = requests.Session.merge_environment_settings

    def patched_merge_environment_settings(self, url, proxies, stream, verify, cert):
        settings = original_merge_environment_settings(
            self, url, proxies, stream, verify, cert
        )
        settings["verify"] = False
        return settings

    requests.Session.merge_environment_settings = patched_merge_environment_settings


def find_dir(base: Path | None, name: str) -> Path | None:
    if base is None or not base.exists():
        return None

    if base.name.upper() == name.upper():
        return base

    for path in base.rglob("*"):
        if path.is_dir() and path.name.upper() == name.upper():
            return path
    return None


def unwrap_single_child(path: Path | None) -> Path | None:
    if path is None or not path.exists():
        return None

    current = path
    while True:
        dirs = [child for child in current.iterdir() if child.is_dir()]
        files = [child for child in current.iterdir() if child.is_file()]
        if len(dirs) != 1 or files:
            return current
        current = dirs[0]


def copy_dir(src: Path | None, dst: Path) -> bool:
    if src is None or not src.exists():
        dst.mkdir(parents=True, exist_ok=True)
        print(f"[WARN] Source not found; empty folder created: {dst.relative_to(REPO_ROOT)}")
        return False

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"[OK] {src} -> {dst.relative_to(REPO_ROOT)}")
    return True


def ensure_images_labels(dataset_dir: Path) -> None:
    for child in ("images", "labels"):
        (dataset_dir / child).mkdir(parents=True, exist_ok=True)


def organize_training_dataset(raw_path: Path) -> None:
    print("\nOrganizing PASCAL_VOC_UAQ_MSUAV...")
    dst = DEST / TRAINING_DATASET
    src = unwrap_single_child(find_dir(raw_path, TRAINING_DATASET))

    if src is None:
        src = raw_path

    copy_dir(find_dir(src, "images"), dst / "images")
    copy_dir(find_dir(src, "labels"), dst / "labels")
    ensure_images_labels(dst)


def organize_evaluation_datasets(raw_path: Path) -> None:
    print("\nOrganizing EVALUATION datasets...")
    eval_dst = DEST / "EVALUATION"
    eval_src = unwrap_single_child(find_dir(raw_path, "EVALUATION"))

    for dataset in EVAL_DATASETS:
        src = unwrap_single_child(find_dir(eval_src, dataset))
        dst = eval_dst / dataset
        copy_dir(src, dst)
        ensure_images_labels(dst)


def organize_evaluation_videos(raw_path: Path) -> None:
    print("\nOrganizing evaluation_videos...")
    videos_dst = DEST / "evaluation_videos"
    videos_dst.mkdir(parents=True, exist_ok=True)

    videos_src = unwrap_single_child(find_dir(raw_path, "evaluation_videos"))
    if videos_src and videos_src.exists():
        subdirs = sorted(path for path in videos_src.iterdir() if path.is_dir())
        for idx, subdir in enumerate(subdirs, start=1):
            copy_dir(subdir, videos_dst / f"VIDEO_{idx}")

    for required in REQUIRED_VIDEO_DIRS:
        (videos_dst / required).mkdir(parents=True, exist_ok=True)


def expected_paths() -> list[Path]:
    return [
        DEST / "evaluation_videos" / "VIDEO_1",
        DEST / "evaluation_videos" / "VIDEO_2",
        DEST / "EVALUATION" / "COCO_TEST" / "images",
        DEST / "EVALUATION" / "COCO_TEST" / "labels",
        DEST / "EVALUATION" / "IRINA" / "images",
        DEST / "EVALUATION" / "IRINA" / "labels",
        DEST / "EVALUATION" / "MANIPAL_UAV" / "images",
        DEST / "EVALUATION" / "MANIPAL_UAV" / "labels",
        DEST / "EVALUATION" / "NTUT" / "images",
        DEST / "EVALUATION" / "NTUT" / "labels",
        DEST / "EVALUATION" / "UAQ_MSUAV_TEST" / "images",
        DEST / "EVALUATION" / "UAQ_MSUAV_TEST" / "labels",
        DEST / "EVALUATION" / "VISDRONE" / "images",
        DEST / "EVALUATION" / "VISDRONE" / "labels",
        DEST / TRAINING_DATASET / "images",
        DEST / TRAINING_DATASET / "labels",
    ]


def print_expected_paths() -> None:
    print("\nExpected relative paths:")
    for path in expected_paths():
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {path.relative_to(REPO_ROOT)}")


def main() -> None:
    configure_kaggle_environment()

    print(f"Downloading Kaggle dataset: {KAGGLE_DATASET}")
    raw_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"Downloaded cache path: {raw_path}")

    DEST.mkdir(parents=True, exist_ok=True)
    organize_training_dataset(raw_path)
    organize_evaluation_datasets(raw_path)
    organize_evaluation_videos(raw_path)
    print_expected_paths()
    print(f"\nDataset ready at: {DEST.resolve()}")


if __name__ == "__main__":
    main()
