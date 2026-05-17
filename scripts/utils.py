"""
UTILS (shared pipeline helpers)
-----------------------------------------------------------------------
Provide shared utilities for:
1) logging and file operations
2) bounding-box geometry and coordinate transforms
3) common helpers reused across the pipeline
-----------------------------------------------------------------------
"""

import json
import logging
import platform
import shutil
import socket
import sqlite3
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Union


def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    Configura un logger robusto que escribe en consola y archivo (opcional).
    Permite trazar la ejecucion del experimento para la tesis.
    """
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def xywh_norm_to_xyxy_abs(box: List[float], img_w: int, img_h: int) -> List[float]:
    """Convierte coordenadas YOLO a absolutas."""
    xc, yc, w, h = box
    w_px = w * img_w
    h_px = h * img_h
    x1 = (xc * img_w) - (w_px / 2)
    y1 = (yc * img_h) - (h_px / 2)
    x2 = (xc * img_w) + (w_px / 2)
    y2 = (yc * img_h) + (h_px / 2)
    return [x1, y1, x2, y2]


def calculate_iou(box_a: List[float], box_b: List[float]) -> float:
    """Calcula Intersection over Union (IoU)."""
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    iou = inter_area / float(box_a_area + box_b_area - inter_area + 1e-6)
    return iou


def ensure_directory(path: Union[str, Path], clean: bool = False):
    """Crea directorios de forma segura."""
    path = Path(path)
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def validate_image_file(file_path: Union[str, Path]) -> bool:
    """Valida existencia y tamano > 0."""
    try:
        p = Path(file_path)
        return p.exists() and p.stat().st_size > 0
    except Exception:
        return False


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    return str(value)


def register_script_run(
    db_path: Union[str, Path],
    script_name: str,
    status: str,
    started_at: datetime,
    finished_at: datetime,
    argv: List[str] | None = None,
    outputs: Dict[str, Any] | None = None,
    extra: Dict[str, Any] | None = None,
    error_text: str | None = None,
) -> None:
    """Registra la ejecucion de un script en una tabla SQLite comun."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS script_run_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                script_name TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                duration_sec REAL NOT NULL,
                cwd TEXT,
                hostname TEXT,
                platform TEXT,
                python_executable TEXT,
                argv_json TEXT,
                outputs_json TEXT,
                extra_json TEXT,
                error_text TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO script_run_registry (
                script_name, status, started_at, finished_at, duration_sec,
                cwd, hostname, platform, python_executable,
                argv_json, outputs_json, extra_json, error_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                script_name,
                status,
                started_at.isoformat(timespec="seconds"),
                finished_at.isoformat(timespec="seconds"),
                max(0.0, (finished_at - started_at).total_seconds()),
                str(Path.cwd()),
                socket.gethostname(),
                platform.platform(),
                sys.executable,
                json.dumps(argv or sys.argv, ensure_ascii=True, default=_json_default),
                json.dumps(outputs or {}, ensure_ascii=True, default=_json_default),
                json.dumps(extra or {}, ensure_ascii=True, default=_json_default),
                error_text,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def run_with_sqlite_registration(
    script_name: str,
    func: Callable[[], Any],
    db_path: Union[str, Path],
    outputs: Dict[str, Any] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Any:
    """Ejecuta una funcion y registra exito o error en SQLite."""
    started_at = datetime.now()
    try:
        result = func()
    except SystemExit as exc:
        finished_at = datetime.now()
        exit_code = exc.code if isinstance(exc.code, int) else 0
        register_script_run(
            db_path=db_path,
            script_name=script_name,
            status="success" if exit_code == 0 else "error",
            started_at=started_at,
            finished_at=finished_at,
            outputs=outputs,
            extra={**(extra or {}), "system_exit_code": exit_code},
            error_text=None if exit_code == 0 else f"SystemExit({exit_code})",
        )
        raise
    except Exception:
        finished_at = datetime.now()
        register_script_run(
            db_path=db_path,
            script_name=script_name,
            status="error",
            started_at=started_at,
            finished_at=finished_at,
            outputs=outputs,
            extra=extra,
            error_text=traceback.format_exc(),
        )
        raise

    finished_at = datetime.now()
    register_script_run(
        db_path=db_path,
        script_name=script_name,
        status="success",
        started_at=started_at,
        finished_at=finished_at,
        outputs=outputs,
        extra=extra,
    )
    return result
