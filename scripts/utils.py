"""
UTILS (shared pipeline helpers)
-----------------------------------------------------------------------
Provide shared utilities for:
1) logging and file operations
2) bounding-box geometry and coordinate transforms
3) common helpers reused across the pipeline
-----------------------------------------------------------------------
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Tuple, Union

def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    Configura un logger robusto que escribe en consola y archivo (opcional).
    Permite trazar la ejecución del experimento para la tesis.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if the module is reloaded
    if not logger.handlers:
        # Console handler (without specific encoding to respect the system console)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler (force UTF-8 to ensure the log file supports everything)
        if log_file:
            # <--- CHANGE HERE: encoding='utf-8'
            fh = logging.FileHandler(log_file, encoding='utf-8') 
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

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    if interArea == 0: return 0.0

    boxAArea = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    boxBArea = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def ensure_directory(path: Union[str, Path], clean: bool = False):
    """Crea directorios de forma segura."""
    path = Path(path)
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def validate_image_file(file_path: Union[str, Path]) -> bool:
    """Valida existencia y tamaño > 0."""
    try:
        p = Path(file_path)
        return p.exists() and p.stat().st_size > 0
    except Exception:
        return False
