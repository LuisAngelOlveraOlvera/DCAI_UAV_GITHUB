"""
02 ANALYSIS ETL (DoE pipeline)
-----------------------------------------------------------------------
Build the analysis metadata database by:
1) indexing images from disk
2) computing pHash values
3) classifying each image source as PASCAL or PROPIO
4) precomputing Hamming-based deduplication groups
-----------------------------------------------------------------------
"""

import sqlite3
import imagehash
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import config
import config_analysis  # Import the analysis-specific configuration
import utils

# ============================================================
# LOGGER
# ============================================================
logger = utils.setup_logger(
    "Analysis_ETL",
    log_file=str(config.LOGS_DIR / "analysis_etl.log")
)

# ============================================================
# 1. INITIALIZE ANALYSIS DB
# ============================================================
def init_analysis_db():
    db_path = config_analysis.ANALYSIS_DB_PATH
    
    if db_path.exists():
        db_path.unlink()
        logger.info(f"BD de análisis anterior eliminada: {db_path.name}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Simplified table for scenario analysis
    # Add 'source' to distinguish PASCAL from PROPIO
    # Add 'judge_score', which will be filled in step 02
    cur.execute("""
    CREATE TABLE analysis_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT NOT NULL,
        relative_path TEXT NOT NULL,
        phash TEXT,
        source TEXT,          -- 'PASCAL' o 'PROPIO'
        judge_score REAL DEFAULT 0.0,
        dedup_cluster INTEGER,
        is_dedup_keep INTEGER DEFAULT 1
    )
    """)
    
    conn.commit()
    return conn

# ============================================================
# 2. FUNCIONES AUXILIARES
# ============================================================
def get_image_source(filename: str) -> str:
    """
    Determina el origen de la imagen basado en los prefijos de config_analysis.
    """
    if filename.startswith(config_analysis.PASCAL_PREFIXES):
        return 'PASCAL'
    return 'PROPIO'

def compute_phash(img_path: Path):
    try:
        with Image.open(img_path) as img:
            # Use the hash size defined in the main config
            h = str(imagehash.phash(img, hash_size=config.HASH_SIZE))
            return h
    except Exception as e:
        logger.error(f"Error procesando {img_path.name}: {e}")
        return None


def apply_hamming_dedup(conn, threshold: int):
    """
    Precompute Hamming-based dedup before Judge stage.
    Marks representative rows with is_dedup_keep=1 and duplicates with 0.
    """
    rows = conn.execute(
        "SELECT id, phash FROM analysis_images WHERE phash IS NOT NULL ORDER BY id"
    ).fetchall()
    total = len(rows)
    if total == 0:
        return 0, 0, 0

    hash_objs = [imagehash.hex_to_hash(r[1]) for r in rows]
    parent = list(range(total))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in tqdm(
        range(total),
        desc=f"Deduplicando Hamming<={threshold}",
        leave=False,
    ):
        for j in range(i + 1, total):
            if (hash_objs[i] - hash_objs[j]) <= threshold:
                union(i, j)

    root_to_cluster = {}
    root_to_keep_id = {}
    next_cluster = 1
    for idx, (img_id, _) in enumerate(rows):
        root = find(idx)
        if root not in root_to_cluster:
            root_to_cluster[root] = next_cluster
            root_to_keep_id[root] = img_id
            next_cluster += 1

    updates = []
    for idx, (img_id, _) in enumerate(rows):
        root = find(idx)
        cluster_id = root_to_cluster[root]
        is_keep = 1 if img_id == root_to_keep_id[root] else 0
        updates.append((cluster_id, is_keep, img_id))

    conn.executemany(
        "UPDATE analysis_images SET dedup_cluster = ?, is_dedup_keep = ? WHERE id = ?",
        updates,
    )
    conn.commit()

    unique_kept = len(root_to_cluster)
    duplicates = total - unique_kept
    return total, unique_kept, duplicates

# ============================================================
# 3. ANALYTICAL INGESTION PROCESS
# ============================================================
def run_analysis_etl():
    logger.info("=== INICIANDO ETL DE ANÁLISIS PRELIMINAR ===")
    
    conn = init_analysis_db()
    cur = conn.cursor()

    # Search all images using the extensions from the main config
    img_files = []
    for ext in config.IMG_EXTENSIONS:
        img_files.extend(config.IMAGES_DIR.rglob(f"*{ext}"))

    logger.info(f"Imágenes detectadas en disco: {len(img_files)}")
    
    # Counters
    stats = {'PASCAL': 0, 'PROPIO': 0, 'ERRORS': 0}
    
    batch_data = []
    BATCH_SIZE = 1000

    for img_path in tqdm(img_files, desc="Clasificando Origen y Hashing"):
        # 1. Compute hash
        phash = compute_phash(img_path)
        if not phash:
            stats['ERRORS'] += 1
            continue

        # 2. Determine source (PASCAL vs PROPIO)
        fname = img_path.name
        source = get_image_source(fname)
        stats[source] += 1
        
        # 3. Portable relative path
        rel_path = img_path.relative_to(config.DATASET_ROOT).as_posix()

        batch_data.append((fname, rel_path, phash, source))

        # Insert in batches for speed
        if len(batch_data) >= BATCH_SIZE:
            cur.executemany("""
                INSERT INTO analysis_images (file_name, relative_path, phash, source)
                VALUES (?, ?, ?, ?)
            """, batch_data)
            batch_data = []

    # Insert remaining rows
    if batch_data:
        cur.executemany("""
            INSERT INTO analysis_images (file_name, relative_path, phash, source)
            VALUES (?, ?, ?, ?)
        """, batch_data)

    conn.commit()
    h_total, h_unique, h_dups = apply_hamming_dedup(conn, config.HAMMING_THRESHOLD)
    h_rate = (h_dups / h_total * 100) if h_total > 0 else 0.0

    # Quick pHash redundancy summary (exact match) for ETL traceability.
    total_indexed = cur.execute("SELECT COUNT(*) FROM analysis_images").fetchone()[0]
    unique_hashes = cur.execute("SELECT COUNT(DISTINCT phash) FROM analysis_images").fetchone()[0]
    duplicates = total_indexed - unique_hashes
    redundancy_rate = (duplicates / total_indexed * 100) if total_indexed > 0 else 0.0
    conn.close()

    # ========================================================
    # REPORTE PRELIMINAR
    # ========================================================
    total = stats['PASCAL'] + stats['PROPIO']
    logger.info("=" * 60)
    logger.info("RESUMEN DE CLASIFICACIÓN DE ORIGEN (FASE 0)")
    logger.info("=" * 60)
    logger.info(f"Total Imágenes Indexadas: {total}")
    logger.info(f"   Fuente PASCAL VOC:    {stats['PASCAL']} ({stats['PASCAL']/total*100:.1f}%)")
    logger.info(f"   Fuente PROPIA:        {stats['PROPIO']} ({stats['PROPIO']/total*100:.1f}%)")
    logger.info(f"   pHash únicos (exact): {unique_hashes}")
    logger.info(f"   pHash duplicados:     {duplicates}")
    logger.info(f"   Redundancia pHash %:  {redundancy_rate:.2f}%")
    logger.info(f"   pHash únicos (Hamming<={config.HAMMING_THRESHOLD}): {h_unique}")
    logger.info(f"   Duplicados (Hamming): {h_dups}")
    logger.info(f"   Redundancia Hamming %:{h_rate:.2f}%")
    logger.info(f"Imágenes corruptas/ilegibles: {stats['ERRORS']}")
    logger.info(f"Metadatos guardados en: {config_analysis.ANALYSIS_DB_PATH}")
    logger.info("=" * 60)
    logger.info("Listo para ejecutar: 07_judge_scoring.py")

if __name__ == "__main__":
    run_analysis_etl()
