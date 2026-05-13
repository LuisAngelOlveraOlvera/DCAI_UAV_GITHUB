"""
04 ETL INGESTION (portable dataset)
-----------------------------------------------------------------------
Build dataset_master.sqlite by:
1) indexing portable image paths
2) reusing pHash values from analysis_metadata.sqlite
3) loading YOLO person annotations
4) summarizing redundancy and ingestion coverage
-----------------------------------------------------------------------
"""

import sqlite3
from PIL import Image
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import config
import config_analysis
import utils

# ============================================================
# LOGGER
# ============================================================
logger = utils.setup_logger(
    "ETL_Ingest",
    log_file=str(config.LOGS_DIR / "etl_ingest.log")
)

# ============================================================
#  INITIALIZE DB (PORTABLE)
# ============================================================
def init_db():
    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
        logger.info("BD anterior eliminada.")

    conn = sqlite3.connect(config.DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT NOT NULL,
        relative_path TEXT NOT NULL UNIQUE,
        width INTEGER,
        height INTEGER,
        phash TEXT,
        label_issue TEXT DEFAULT 'unknown'
    )
    """)

    cur.execute("""
    CREATE TABLE annotations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER,
        class_id INTEGER,
        x_center REAL,
        y_center REAL,
        width REAL,
        height REAL,
        FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
    )
    """)
    # Contract for downstream QA:
    # x_center, y_center, width, height are stored in normalized YOLO format.
    # Geometric checks convert these values to absolute xyxy via utils.py.

    conn.commit()
    return conn

# ============================================================
# PHASH REUSE FROM 02_analysis_etl.py
# ============================================================
def load_phash_index():
    db_path = config_analysis.ANALYSIS_DB_PATH
    if not db_path.exists():
        raise FileNotFoundError(
            f"No existe {db_path}. Ejecuta primero: python 02_analysis_etl.py"
        )

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT relative_path, phash FROM analysis_images WHERE phash IS NOT NULL",
            conn,
        )
    finally:
        conn.close()

    return dict(zip(df["relative_path"], df["phash"]))


def get_image_size(img_path: Path):
    try:
        with Image.open(img_path) as img:
            return img.size
    except Exception as e:
        logger.error(f"Error en imagen {img_path.name}: {e}")
        return None

# ============================================================
# BUSCAR LABEL (PORTABLE)
# ============================================================
def find_label_path(img_path: Path):
    label_path = config.LABELS_DIR / f"{img_path.stem}.txt"
    return label_path if label_path.exists() else None

# ============================================================
# REDUNDANCY ANALYSIS
# ============================================================
def load_hamming_summary():
    """
    Reuse dedup summary already computed in 02_analysis_etl.py.
    """
    db_path = config_analysis.ANALYSIS_DB_PATH
    if not db_path.exists():
        return None

    conn = sqlite3.connect(db_path)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(analysis_images)").fetchall()]
        if "is_dedup_keep" not in cols:
            return None

        total = conn.execute(
            "SELECT COUNT(*) FROM analysis_images WHERE phash IS NOT NULL"
        ).fetchone()[0]
        unique = conn.execute(
            "SELECT COUNT(*) FROM analysis_images WHERE is_dedup_keep = 1"
        ).fetchone()[0]
        dups = max(total - unique, 0)
        rate = (dups / total * 100) if total > 0 else 0.0
        return total, unique, dups, rate
    finally:
        conn.close()

# ============================================================
# ETL PRINCIPAL
# ============================================================
def run_etl():
    logger.info("=== INICIANDO ETL PORTABLE ===")
    conn = init_db()
    cur = conn.cursor()
    try:
        phash_index = load_phash_index()
    except Exception as e:
        logger.critical(f"No se pudo cargar índice pHash: {e}")
        conn.close()
        return

    img_files = []
    for ext in config.IMG_EXTENSIONS:
        img_files.extend(config.IMAGES_DIR.rglob(f"*{ext}"))

    logger.info(f"Imágenes encontradas: {len(img_files)}")

    counters = {"images": 0, "annotations": 0}
    missing_labels = 0
    missing_phash = 0

    for img_path in tqdm(img_files, desc="ETL"):
        size = get_image_size(img_path)
        if not size:
            continue

        w, h = size

        # PORTABLE RELATIVE PATH
        relative_path = img_path.relative_to(config.DATASET_ROOT).as_posix()
        phash = phash_index.get(relative_path)
        if not phash:
            missing_phash += 1
            continue

        try:
            cur.execute("""
                INSERT INTO images (file_name, relative_path, width, height, phash)
                VALUES (?, ?, ?, ?, ?)
            """, (img_path.name, relative_path, w, h, phash))

            image_id = cur.lastrowid
            counters["images"] += 1

            label_path = find_label_path(img_path)

            if label_path:
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        # Keep only class 0 (person) with YOLO normalized fields.
                        if len(parts) >= 5 and int(parts[0]) == 0:
                            cls, xc, yc, bw, bh = map(float, parts[:5])
                            cur.execute("""
                                INSERT INTO annotations
                                (image_id, class_id, x_center, y_center, width, height)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (image_id, int(cls), xc, yc, bw, bh))
                            counters["annotations"] += 1
            else:
                missing_labels += 1

        except sqlite3.IntegrityError:
            continue

    conn.commit()

    # ========================================================
    # REPORTE
    # ========================================================
    summary = load_hamming_summary()
    if summary is None:
        total = counters["images"]
        unique = counters["images"]
        dups = 0
        rate = 0.0
        logger.warning(
            "No se encontró resumen Hamming precomputado; usando conteos de ingesta."
        )
    else:
        total, unique, dups, rate = summary

    logger.info("=" * 55)
    logger.info("REPORTE DE INGENIERÍA DE DATOS (OBJ 1)")
    logger.info("=" * 55)
    logger.info(f"Imágenes totales: {total}")
    logger.info(f"Imágenes únicas (Hamming<={config.HAMMING_THRESHOLD}): {unique}")
    logger.info(f"Duplicados: {dups}")
    logger.info(f"Tasa de redundancia: {rate:.2f}%")
    logger.info(f"Etiquetas (clase 0): {counters['annotations']}")
    logger.info(f"Imágenes sin label: {missing_labels}")
    logger.info(f"Imágenes sin pHash reutilizable: {missing_phash}")
    logger.info("=" * 55)

    conn.close()
    logger.info("ETL finalizado correctamente.")

# ============================================================
if __name__ == "__main__":
    run_etl()
