"""
CONFIG (physical pipeline settings)
-----------------------------------------------------------------------
Central configuration for:
1) dataset paths and output folders
2) split ratios and image extensions
3) runtime thresholds and infrastructure constants
-----------------------------------------------------------------------
"""

# ============================================================
# config.py - INFRASTRUCTURE CONFIGURATION (PHYSICAL)
# ============================================================
from pathlib import Path

# 1. SYSTEM PATHS
DATASET_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DATASET_DIR = DATASET_ROOT / "DATASET_KAGGLE"

IMAGES_DIR  = SOURCE_DATASET_DIR / "images"
LABELS_DIR  = SOURCE_DATASET_DIR / "labels"
LOGS_DIR    = DATASET_ROOT / "logs"
EXPORTS_DIR = DATASET_ROOT / "exports"

# Create folders if they do not exist
for d in [LOGS_DIR, EXPORTS_DIR]:
    d.mkdir(exist_ok=True, parents=True)

# 2. ARCHIVOS GLOBALES
DB_PATH = DATASET_ROOT / "dataset_master.sqlite"
MODEL_JUDGE_PATH = "yolo11x.pt"

# 3. CONSTANTES GLOBALES
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
HASH_SIZE = 8
HAMMING_THRESHOLD = 1
SEED = 0
IOU_THRESHOLD = 0.5

# 4. COMPATIBILITY AND SPLIT PARAMETERS
OUTPUT_DIR = LOGS_DIR
CONF_THRESHOLD = 0.35 

# ===> THIS WAS THE MISSING LINE <===
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # 80% Train, 10% Val, 10% Test
