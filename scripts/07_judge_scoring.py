"""
07 JUDGE SCORING (DoE pipeline)
-----------------------------------------------------------------------
Run the judge model once per image to:
1) store judge_score values for DoE thresholds
2) classify semantic QA outcomes in label_issue
3) persist progress for resumable processing
-----------------------------------------------------------------------
"""

import gc
import sqlite3
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from ultralytics import YOLO

import config
import config_analysis
import utils

logger = utils.setup_logger(
    "Judge_OnePass",
    log_file=str(config.LOGS_DIR / "judge_onepass.log"),
)


REQUIRED_COLUMNS = {
    "processed": "INTEGER DEFAULT 0",
    "label_issue": "TEXT DEFAULT 'unknown'",
    "judge_has_pred": "INTEGER DEFAULT 0",
    "judge_pred_count": "INTEGER DEFAULT 0",
    "judge_max_iou": "REAL DEFAULT 0.0",
}


def has_column(conn, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


def ensure_analysis_columns(conn):
    for col, ddl in REQUIRED_COLUMNS.items():
        if not has_column(conn, "analysis_images", col):
            conn.execute(f"ALTER TABLE analysis_images ADD COLUMN {col} {ddl}")
    conn.commit()


def load_ground_truth_index() -> dict[str, list[list[float]]]:
    """Build GT boxes index by relative_path from dataset_master.sqlite."""
    if not config.DB_PATH.exists():
        logger.error(f"No existe DB maestra: {config.DB_PATH}")
        return {}

    conn = sqlite3.connect(config.DB_PATH)
    try:
        images_rows = conn.execute(
            "SELECT id, relative_path, width, height FROM images"
        ).fetchall()
        ann_rows = conn.execute(
            "SELECT image_id, x_center, y_center, width, height FROM annotations"
        ).fetchall()
    finally:
        conn.close()

    dims_by_image_id = {
        img_id: (rel_path, w, h) for img_id, rel_path, w, h in images_rows
    }

    gt_by_path = {}
    for img_id, rel_path, w, h in images_rows:
        gt_by_path[rel_path] = []

    for image_id, xc, yc, bw, bh in ann_rows:
        dims = dims_by_image_id.get(image_id)
        if not dims:
            continue
        rel_path, w, h = dims
        box = utils.xywh_norm_to_xyxy_abs([xc, yc, bw, bh], w, h)
        gt_by_path.setdefault(rel_path, []).append(box)

    logger.info(f"GT indexado: {len(gt_by_path)} imágenes")
    return gt_by_path


def classify_label_issue(gt_boxes: list[list[float]], pred_boxes: list[list[float]], iou_threshold: float) -> tuple[str, float]:
    has_gt = len(gt_boxes) > 0
    has_pred = len(pred_boxes) > 0

    if has_pred and not has_gt:
        return "missing_label", 0.0

    if has_gt and not has_pred:
        return "bad_label", 0.0

    if has_gt and has_pred:
        max_iou_img = 0.0
        for gt in gt_boxes:
            local_max = 0.0
            for pred in pred_boxes:
                iou = utils.calculate_iou(gt, pred)
                if iou > local_max:
                    local_max = iou
            if local_max > max_iou_img:
                max_iou_img = local_max

        if max_iou_img < iou_threshold:
            return "poor_alignment", max_iou_img
        return "ok", max_iou_img

    return "ok", 0.0


def propagate_to_duplicates(conn):
    """Copy representative judge outputs to duplicate rows in same dedup cluster."""
    conn.execute(
        """
        UPDATE analysis_images AS d
        SET judge_score = (
            SELECT k.judge_score
            FROM analysis_images AS k
            WHERE k.dedup_cluster = d.dedup_cluster AND k.is_dedup_keep = 1
            LIMIT 1
        ),
        label_issue = (
            SELECT k.label_issue
            FROM analysis_images AS k
            WHERE k.dedup_cluster = d.dedup_cluster AND k.is_dedup_keep = 1
            LIMIT 1
        ),
        judge_has_pred = (
            SELECT k.judge_has_pred
            FROM analysis_images AS k
            WHERE k.dedup_cluster = d.dedup_cluster AND k.is_dedup_keep = 1
            LIMIT 1
        ),
        judge_pred_count = (
            SELECT k.judge_pred_count
            FROM analysis_images AS k
            WHERE k.dedup_cluster = d.dedup_cluster AND k.is_dedup_keep = 1
            LIMIT 1
        ),
        judge_max_iou = (
            SELECT k.judge_max_iou
            FROM analysis_images AS k
            WHERE k.dedup_cluster = d.dedup_cluster AND k.is_dedup_keep = 1
            LIMIT 1
        ),
        processed = CASE WHEN d.processed = 0 THEN 1 ELSE d.processed END
        WHERE d.is_dedup_keep = 0 AND d.dedup_cluster IS NOT NULL
        """
    )
    conn.commit()


def run_judge_onepass():
    logger.info("=== INICIANDO JUDGE ONE-PASS (SCORING + QA) ===")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA disponible: Sí | GPU: {gpu_name}")
        infer_device = 0
    else:
        logger.warning("CUDA disponible: No | Se ejecutará en CPU")
        infer_device = "cpu"

    iou_threshold = getattr(config, "IOU_THRESHOLD", 0.5)
    logger.info(f"IOU_THRESHOLD={iou_threshold} | CONF_THRESHOLD={config.CONF_THRESHOLD}")

    model_path = Path(config_analysis.MODEL_JUDGE_PATH)
    if not model_path.exists():
        model_path = Path("yolo11x.pt")

    try:
        model = YOLO(str(model_path))
        logger.info(f"Modelo cargado: {model_path}")
    except Exception as e:
        logger.critical(f"Error cargando modelo juez: {e}")
        sys.exit(1)

    gt_by_path = load_ground_truth_index()
    if not gt_by_path:
        logger.critical("No hay GT disponible en DB maestra; no se puede auditar en una sola pasada.")
        sys.exit(1)

    conn = sqlite3.connect(config_analysis.ANALYSIS_DB_PATH)
    ensure_analysis_columns(conn)
    cur = conn.cursor()

    has_dedup_keep = has_column(conn, "analysis_images", "is_dedup_keep")
    if has_dedup_keep:
        rows = cur.execute(
            """
            SELECT id, relative_path, file_name
            FROM analysis_images
            WHERE processed = 0 AND is_dedup_keep = 1
            """
        ).fetchall()
    else:
        rows = cur.execute(
            """
            SELECT id, relative_path, file_name
            FROM analysis_images
            WHERE processed = 0
            """
        ).fetchall()

    total_pending = len(rows)
    if total_pending == 0:
        logger.info("No hay pendientes. Si necesitas recalcular, resetea processed=0.")
        conn.close()
        return

    logger.info(f"Pendientes: {total_pending}")
    pbar = tqdm(total=total_pending, desc="Judge one-pass")

    updates = []
    batch_commit = 10
    stats = {"ok": 0, "missing_label": 0, "bad_label": 0, "poor_alignment": 0, "unknown": 0}

    for img_id, rel_path, fname in rows:
        img_path = config.DATASET_ROOT / rel_path

        score = 0.0
        processed = 1
        label_issue = "unknown"
        has_pred = 0
        pred_count = 0
        max_iou_img = 0.0

        try:
            if not img_path.exists():
                logger.warning(f"Archivo faltante: {rel_path}")
                processed = 2
            else:
                results = model.predict(
                    source=str(img_path),
                    classes=[0],
                    conf=config_analysis.INFERENCE_CONF_THRESHOLD,
                    verbose=False,
                    device=infer_device,
                )

                pred_boxes = []
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    confs = boxes.conf.detach().cpu().tolist()
                    xyxy = boxes.xyxy.detach().cpu().tolist()
                    score = max(confs)
                    for conf, box in zip(confs, xyxy):
                        if conf >= config.CONF_THRESHOLD:
                            pred_boxes.append(box)

                pred_count = len(pred_boxes)
                has_pred = 1 if pred_count > 0 else 0

                gt_boxes = gt_by_path.get(rel_path, [])
                label_issue, max_iou_img = classify_label_issue(gt_boxes, pred_boxes, iou_threshold)

        except Exception as e:
            logger.error(f"Error en {fname}: {e}")
            processed = 2

        stats[label_issue] = stats.get(label_issue, 0) + 1
        updates.append((score, processed, label_issue, has_pred, pred_count, max_iou_img, img_id))

        if len(updates) >= batch_commit:
            cur.executemany(
                """
                UPDATE analysis_images
                SET judge_score = ?,
                    processed = ?,
                    label_issue = ?,
                    judge_has_pred = ?,
                    judge_pred_count = ?,
                    judge_max_iou = ?
                WHERE id = ?
                """,
                updates,
            )
            conn.commit()
            updates = []

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        pbar.update(1)

    if updates:
        cur.executemany(
            """
            UPDATE analysis_images
            SET judge_score = ?,
                processed = ?,
                label_issue = ?,
                judge_has_pred = ?,
                judge_pred_count = ?,
                judge_max_iou = ?
            WHERE id = ?
            """,
            updates,
        )
        conn.commit()

    if has_dedup_keep:
        propagate_to_duplicates(conn)

    pbar.close()
    conn.close()

    logger.info("=" * 60)
    logger.info("RESUMEN QA (ONE-PASS)")
    logger.info(f"ok={stats.get('ok', 0)}")
    logger.info(f"missing_label={stats.get('missing_label', 0)}")
    logger.info(f"bad_label={stats.get('bad_label', 0)}")
    logger.info(f"poor_alignment={stats.get('poor_alignment', 0)}")
    logger.info(f"unknown={stats.get('unknown', 0)}")
    logger.info("Listo para: python 08_qa_audit.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_judge_onepass()
