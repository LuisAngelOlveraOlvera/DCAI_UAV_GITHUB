import gc
import os
import queue
import re
import threading
import time
import zipfile
from collections import defaultdict, deque
from pathlib import Path
from xml.sax.saxutils import escape

import cv2 as cv
import numpy as np
import pandas as pd
from ultralytics import YOLO

from KalmanTracker import KalmanTracker
from config import DATASET_ROOT, EVALUATION_VIDEOS_DIR, EXPORTS_DIR


DETECT_EVERY = 3
CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.3
MAX_DETECTIONS = 12
KALMAN_MAX_AGE = 10
KALMAN_MIN_HITS = 3
KALMAN_IOU_THRESHOLD = 0.3
EVAL_IOU_THRESHOLD = 0.25
MAP_IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg")
WEIGHTS_BASE_PATH = DATASET_ROOT / "runs" / "train"
OUTPUT_DIR = EXPORTS_DIR / "reportes_kalman_individual"


try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        print(f"[INFO] CUDA disponible: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] CUDA no disponible. Se usara CPU.")
except Exception as exc:
    torch = None
    DEVICE = "cpu"
    print(f"[WARN] No se pudo inicializar torch/CUDA: {exc}")

cv.setNumThreads(4)
gc.set_threshold(700, 10, 10)

RESOLUCIONES = {
    "1": ("256p", (256, 256)),
    "2": ("340p", (340, 340)),
    "3": ("480p", (640, 480)),
    "4": ("720p", (1280, 720)),
    "5": ("1080p", (1920, 1080)),
    "6": ("Original", None),
}


def natural_key(text):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(text))]


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def excel_col_name(col_idx):
    name = ""
    while col_idx:
        col_idx, remainder = divmod(col_idx - 1, 26)
        name = chr(65 + remainder) + name
    return name


def excel_cell(value, row_idx, col_idx):
    cell_ref = f"{excel_col_name(col_idx)}{row_idx}"
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f'<c r="{cell_ref}"/>'
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return f'<c r="{cell_ref}"><v>{float(value)}</v></c>'
    return f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def dataframe_to_sheet_xml(df):
    rows = []
    headers = list(df.columns)
    header_cells = "".join(excel_cell(header, 1, col_idx) for col_idx, header in enumerate(headers, start=1))
    rows.append(f'<row r="1">{header_cells}</row>')
    for row_idx, (_, row) in enumerate(df.iterrows(), start=2):
        cells = "".join(excel_cell(row[header], row_idx, col_idx) for col_idx, header in enumerate(headers, start=1))
        rows.append(f'<row r="{row_idx}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(rows)}</sheetData>'
        "</worksheet>"
    )


def write_xlsx_with_tabs(path, sheets):
    sheet_names = list(sheets.keys())
    workbook_sheets = "".join(
        f'<sheet name="{escape(name[:31])}" sheetId="{idx}" r:id="rId{idx}"/>'
        for idx, name in enumerate(sheet_names, start=1)
    )
    workbook_rels = "".join(
        f'<Relationship Id="rId{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{idx}.xml"/>'
        for idx in range(1, len(sheet_names) + 1)
    )
    workbook_rels += (
        f'<Relationship Id="rId{len(sheet_names) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    content_types_sheets = "".join(
        f'<Override PartName="/xl/worksheets/sheet{idx}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for idx in range(1, len(sheet_names) + 1)
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as xlsx:
        xlsx.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            f"{content_types_sheets}"
            "</Types>",
        )
        xlsx.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        xlsx.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f"<sheets>{workbook_sheets}</sheets>"
            "</workbook>",
        )
        xlsx.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{workbook_rels}"
            "</Relationships>",
        )
        xlsx.writestr(
            "xl/styles.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
            "</styleSheet>",
        )
        for idx, name in enumerate(sheet_names, start=1):
            xlsx.writestr(f"xl/worksheets/sheet{idx}.xml", dataframe_to_sheet_xml(sheets[name]))


def discover_weights():
    runs = []
    if not WEIGHTS_BASE_PATH.exists():
        return runs
    for run_dir in sorted((p for p in WEIGHTS_BASE_PATH.iterdir() if p.is_dir()), key=lambda p: natural_key(p.name)):
        weight_path = run_dir / "weights" / "best.pt"
        if weight_path.exists():
            runs.append((run_dir.name, weight_path))
    return runs


def discover_labels_directories():
    candidates = []
    roots = [EVALUATION_VIDEOS_DIR, DATASET_ROOT / "EVALUATION", DATASET_ROOT / "DATASET_KAGGLE"]
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for label_dir in root.rglob("labels"):
            if not label_dir.is_dir():
                continue
            has_frames = any(label_dir.glob("frame_*.txt"))
            if has_frames and str(label_dir) not in seen:
                seen.add(str(label_dir))
                candidates.append(label_dir)
    return sorted(candidates, key=lambda p: natural_key(p))


def discover_videos():
    videos = []
    roots = [EVALUATION_VIDEOS_DIR, DATASET_ROOT / "EVALUATION", DATASET_ROOT / "DATASET_KAGGLE"]
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for ext in VIDEO_EXTENSIONS:
            for video_path in root.rglob(f"*{ext}"):
                if str(video_path) not in seen:
                    seen.add(str(video_path))
                    videos.append(video_path)
    return sorted(videos, key=lambda p: natural_key(p))


def choose_from_menu(title, options, formatter):
    if not options:
        raise FileNotFoundError(f"No hay opciones disponibles para: {title}")
    print(f"\n{title}")
    print("-" * len(title))
    for idx, item in enumerate(options, start=1):
        print(f"{idx}. {formatter(item)}")
    while True:
        raw = input("Selecciona una opcion por numero: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print("[WARN] Opcion invalida. Intenta de nuevo.")


def choose_resolution():
    print("\nResoluciones disponibles")
    print("------------------------")
    for key, (label, size) in RESOLUCIONES.items():
        suffix = " (nativa)" if size is None else f" ({size[0]}x{size[1]})"
        print(f"{key}. {label}{suffix}")
    while True:
        choice = input("Selecciona la resolucion [6]: ").strip() or "6"
        if choice in RESOLUCIONES:
            return RESOLUCIONES[choice]
        print("[WARN] Opcion invalida. Intenta de nuevo.")


def yolo_to_xyxy(values, frame_w, frame_h):
    _, cx, cy, w, h = map(float, values[:5])
    box_w, box_h = w * frame_w, h * frame_h
    return [
        (cx * frame_w) - (box_w / 2),
        (cy * frame_h) - (box_h / 2),
        (cx * frame_w) + (box_w / 2),
        (cy * frame_h) + (box_h / 2),
    ]


def load_ground_truth(labels_path, frame_idx, frame_w, frame_h):
    label_file = labels_path / f"frame_{frame_idx:06d}.txt"
    if not label_file.exists():
        return []
    boxes = []
    with open(label_file, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append(yolo_to_xyxy(parts, frame_w, frame_h))
    return boxes


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def match_counts(gt_by_frame, pred_by_frame, frame_indices, iou_threshold):
    tp = fp = fn = tn = 0
    for frame_idx in frame_indices:
        gt_boxes = gt_by_frame.get(frame_idx, [])
        pred_boxes = pred_by_frame.get(frame_idx, [])
        matched_gt = set()
        for pred in sorted(pred_boxes, key=lambda item: item["score"], reverse=True):
            best_iou, best_gt_idx = 0.0, None
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = bbox_iou(pred["bbox"], gt_box)
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, gt_idx
            if best_gt_idx is not None and best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        fn += max(0, len(gt_boxes) - len(matched_gt))
        if not gt_boxes and not pred_boxes:
            tn += 1
    return tp, tn, fp, fn


def frame_iou_rows(gt_by_frame, pred_by_frame, frame_indices, iou_threshold):
    rows = []
    for frame_idx in sorted(frame_indices):
        gt_boxes = gt_by_frame.get(frame_idx, [])
        pred_boxes = sorted(pred_by_frame.get(frame_idx, []), key=lambda item: item["score"], reverse=True)
        matched_gt = set()
        matched_ious = []
        all_best_ious = []
        tp = fp = 0
        for pred in pred_boxes:
            best_iou, best_gt_idx = 0.0, None
            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = bbox_iou(pred["bbox"], gt_box)
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, gt_idx
            all_best_ious.append(best_iou)
            if best_gt_idx is not None and best_gt_idx not in matched_gt and best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
                matched_ious.append(best_iou)
            else:
                fp += 1
        fn = max(0, len(gt_boxes) - len(matched_gt))
        rows.append(
            {
                "Frame": frame_idx,
                "GT Anotaciones": len(gt_boxes),
                "Predicciones Modelo": len(pred_boxes),
                "TP": tp,
                "TN": 1 if not gt_boxes and not pred_boxes else 0,
                "FP": fp,
                "FN": fn,
                "IoU Promedio Matches": float(np.mean(matched_ious)) if matched_ious else 0.0,
                "IoU Maximo Match": float(np.max(matched_ious)) if matched_ious else 0.0,
                "IoU Promedio Predicciones": float(np.mean(all_best_ious)) if all_best_ious else 0.0,
                "IoU Maximo Prediccion": float(np.max(all_best_ious)) if all_best_ious else 0.0,
                "Umbral IoU": iou_threshold,
            }
        )
    return rows


def classify_detections_for_display(detections, gt_boxes, iou_threshold):
    matched_gt = set()
    classified = []
    for detection in detections:
        best_iou, best_gt_idx = 0.0, None
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = bbox_iou(detection["bbox"], gt_box)
            if iou > best_iou:
                best_iou, best_gt_idx = iou, gt_idx
        is_true_positive = best_gt_idx is not None and best_iou >= iou_threshold
        if is_true_positive:
            matched_gt.add(best_gt_idx)
        classified.append(
            {
                "bbox": detection["bbox"],
                "track_id": detection["track_id"],
                "score": detection["score"],
                "is_true_positive": is_true_positive,
                "iou": best_iou,
            }
        )
    return classified


def average_precision(gt_by_frame, prediction_records, frame_indices, iou_threshold):
    total_gt = sum(len(gt_by_frame.get(frame_idx, [])) for frame_idx in frame_indices)
    if total_gt == 0:
        return 0.0
    matched = {frame_idx: set() for frame_idx in frame_indices}
    tp_values = []
    fp_values = []
    for pred in sorted(prediction_records, key=lambda item: item["score"], reverse=True):
        frame_idx = pred["frame_idx"]
        if frame_idx not in matched:
            continue
        best_iou, best_gt_idx = 0.0, None
        for gt_idx, gt_box in enumerate(gt_by_frame.get(frame_idx, [])):
            if gt_idx in matched[frame_idx]:
                continue
            iou = bbox_iou(pred["bbox"], gt_box)
            if iou > best_iou:
                best_iou, best_gt_idx = iou, gt_idx
        if best_gt_idx is not None and best_iou >= iou_threshold:
            tp_values.append(1)
            fp_values.append(0)
            matched[frame_idx].add(best_gt_idx)
        else:
            tp_values.append(0)
            fp_values.append(1)
    if not tp_values:
        return 0.0
    tp_cum = np.cumsum(tp_values)
    fp_cum = np.cumsum(fp_values)
    recalls = tp_cum / total_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    ap = 0.0
    for recall_threshold in np.linspace(0, 1, 101):
        valid = precisions[recalls >= recall_threshold]
        ap += np.max(valid) if valid.size else 0.0
    return ap / 101


def compute_detection_report(gt_by_frame, pred_by_frame, prediction_records, frame_indices):
    frame_indices = sorted(frame_indices)
    tp, tn, fp, fn = match_counts(gt_by_frame, pred_by_frame, frame_indices, EVAL_IOU_THRESHOLD)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    map50 = average_precision(gt_by_frame, prediction_records, frame_indices, 0.50)
    map5095 = float(np.mean([average_precision(gt_by_frame, prediction_records, frame_indices, threshold) for threshold in MAP_IOU_THRESHOLDS]))
    return {
        "F1": f1,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "MAP5095": map5095,
        "MAP50": map50,
    }


def default_video_for_labels(labels_path, videos):
    parent = labels_path.parent
    same_parent = [video for video in videos if video.parent == parent]
    if same_parent:
        return same_parent
    return videos


def build_output_path(run_name, video_path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{run_name}_{video_path.stem}")
    return OUTPUT_DIR / f"{safe_name}.xlsx"


def main():
    runs = discover_weights()
    labels_dirs = discover_labels_directories()
    videos = discover_videos()

    if not runs:
        raise FileNotFoundError(f"No se encontraron pesos en: {WEIGHTS_BASE_PATH}")
    if not labels_dirs:
        raise FileNotFoundError("No se encontraron carpetas labels con archivos frame_*.txt.")
    if not videos:
        raise FileNotFoundError("No se encontraron videos en las rutas exploradas del dataset.")

    run_name, model_path = choose_from_menu("Pesos disponibles en runs\\train", runs, lambda item: item[0])
    labels_path = choose_from_menu("Carpetas de labels disponibles", labels_dirs, lambda item: str(item))
    candidate_videos = default_video_for_labels(labels_path, videos)
    video_path = choose_from_menu("Videos disponibles", candidate_videos, lambda item: str(item))
    resolution_name, target_size = choose_resolution()
    save_path = build_output_path(run_name, video_path)

    print("\n[INFO] Configuracion seleccionada")
    print(f"Peso: {model_path}")
    print(f"Labels: {labels_path}")
    print(f"Video: {video_path}")
    print(f"Resolucion: {resolution_name}")
    print(f"Salida: {save_path}")

    print(f"[INFO] Backend: {DEVICE.upper()} - Cargando modelo Kalman con DETECT_EVERY = {DETECT_EVERY}...")
    model = YOLO(str(model_path)).to(DEVICE)
    person_class = next((k for k, v in model.names.items() if v == "person"), 0)
    kalman_tracker = KalmanTracker(max_age=KALMAN_MAX_AGE, min_hits=KALMAN_MIN_HITS, iou_threshold=KALMAN_IOU_THRESHOLD)
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    if target_size is None:
        new_width, new_height = video_width, video_height
    else:
        new_width, new_height = target_size
    print(f"[INFO] Video listo. Total Frames: {total_frames}")

    start_time = time.time()
    processed_frames = 0
    total_person_frames = 0
    display_frame_count = 0
    inference_times = deque(maxlen=200)
    latency_times = deque(maxlen=200)
    person_confidences = deque(maxlen=200)
    gt_by_frame = {}
    pred_by_frame = defaultdict(list)
    prediction_records = []
    evaluated_frame_indices = set()

    fallos_continuidad = 0
    last_track_centers = {}
    box_displacements = []
    frame_queue = queue.Queue(maxsize=20)
    result_queue = queue.Queue(maxsize=20)
    stop_event = threading.Event()
    lock = threading.Lock()
    paused = False

    def capture_thread():
        frame_idx = 0
        while not stop_event.is_set():
            if paused:
                time.sleep(0.01)
                continue
            ret, frame = cap.read()
            if not ret:
                break
            current_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            current_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            if target_size is not None and (new_width, new_height) != (current_w, current_h):
                frame = cv.resize(frame, (new_width, new_height), interpolation=cv.INTER_LINEAR)
            try:
                frame_queue.put((frame_idx, frame), timeout=0.1)
                frame_idx += 1
            except queue.Full:
                if stop_event.is_set():
                    break
                time.sleep(0.01)
        stop_event.set()

    def process_thread():
        nonlocal processed_frames, total_person_frames
        while not stop_event.is_set():
            if paused:
                time.sleep(0.01)
                continue
            try:
                frame_idx, frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue

            thread_start_time = time.time()
            tracked_detections = []

            if frame_idx % DETECT_EVERY == 0:
                with lock:
                    processed_frames += 1
                inference_start = time.time()
                frame_h, frame_w = frame.shape[:2]
                detections_for_kalman = []
                score_by_index = {}

                results = model.predict(
                    frame,
                    conf=CONF_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    classes=[person_class],
                    verbose=False,
                    max_det=MAX_DETECTIONS,
                    device=DEVICE,
                    imgsz=640,
                )
                with lock:
                    inference_times.append((time.time() - inference_start) * 1000)

                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    with lock:
                        total_person_frames += 1
                    xyxy_list = boxes.xyxy.cpu().numpy().tolist()
                    scores = boxes.conf.cpu().numpy().tolist()
                    for idx, (bbox, score) in enumerate(zip(xyxy_list, scores)):
                        x1, y1, x2, y2 = bbox
                        detections_for_kalman.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, score, person_class])
                        score_by_index[idx] = float(score)
                    with lock:
                        person_confidences.extend(scores)

                tracks = kalman_tracker.update(detections_for_kalman)
                for track in tracks:
                    x, y, w, h, track_id, _, _ = track
                    bbox = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
                    tracked_detections.append(
                        {
                            "frame_idx": frame_idx,
                            "bbox": bbox,
                            "score": score_by_index.get(len(tracked_detections), 1.0),
                            "track_id": int(track_id),
                        }
                    )

                with lock:
                    gt_by_frame[frame_idx] = load_ground_truth(labels_path, frame_idx, frame_w, frame_h)
                    pred_by_frame[frame_idx] = [{"bbox": item["bbox"], "score": item["score"]} for item in tracked_detections]
                    prediction_records.extend(tracked_detections)
                    evaluated_frame_indices.add(frame_idx)
            else:
                tracks = kalman_tracker.update([])
                for track in tracks:
                    x, y, w, h, track_id, _, _ = track
                    tracked_detections.append(
                        {
                            "frame_idx": frame_idx,
                            "bbox": [x - w / 2, y - h / 2, x + w / 2, y + h / 2],
                            "score": 1.0,
                            "track_id": int(track_id),
                        }
                    )

            try:
                result_queue.put({"frame": frame, "detections": tracked_detections, "frame_idx": frame_idx}, timeout=0.1)
            except queue.Full:
                if stop_event.is_set():
                    break
                time.sleep(0.01)

            with lock:
                latency_times.append((time.time() - thread_start_time) * 1000)

    window_name = "Paralelo c/Kalman - Reporte Metricas"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    thread_capture = threading.Thread(target=capture_thread)
    thread_process = threading.Thread(target=process_thread)
    thread_capture.start()
    thread_process.start()

    habia_deteccion_previa = False
    while not stop_event.is_set() or not result_queue.empty():
        try:
            payload = result_queue.get(timeout=0.1)
            frame = payload["frame"]
            detections = payload["detections"]
            frame_idx = payload["frame_idx"]
        except queue.Empty:
            if stop_event.is_set():
                break
            continue

        if not paused:
            display_frame_count += 1
            hay_deteccion_actual = len(detections) > 0
            if display_frame_count > 1 and habia_deteccion_previa and not hay_deteccion_actual:
                with lock:
                    fallos_continuidad += 1
            habia_deteccion_previa = hay_deteccion_actual

            current_track_ids = set()
            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]
                track_id = detection["track_id"]
                current_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                current_track_ids.add(track_id)
                if track_id in last_track_centers:
                    with lock:
                        box_displacements.append(np.linalg.norm(current_center - last_track_centers[track_id]))
                last_track_centers[track_id] = current_center
            extinct_tracks = set(last_track_centers.keys()) - current_track_ids
            for track_id in extinct_tracks:
                del last_track_centers[track_id]

            frame_gt_boxes = gt_by_frame.get(frame_idx, load_ground_truth(labels_path, frame_idx, frame.shape[1], frame.shape[0]))
            classified_detections = classify_detections_for_display(detections, frame_gt_boxes, EVAL_IOU_THRESHOLD)

            annotated = frame.copy()
            for item in classified_detections:
                x1, y1, x2, y2 = map(int, item["bbox"])
                color = (0, 255, 0) if item["is_true_positive"] else (0, 0, 255)
                label = "TP" if item["is_true_positive"] else "FP"
                cv.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv.putText(
                    annotated,
                    f"ID:{item['track_id']} {label} IoU:{item['iou']:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            elapsed = time.time() - start_time
            display_fps = display_frame_count / elapsed if elapsed > 0 else 0
            with lock:
                avg_latency = np.mean(latency_times) if latency_times else 0
                avg_inference = np.mean(inference_times) if inference_times else 0
                avg_confidence = np.mean(person_confidences) if person_confidences else 0
                porcentaje = safe_div(total_person_frames, processed_frames) * 100
                indice_inestabilidad = safe_div(fallos_continuidad, display_frame_count) * 100
                dsd_running = np.std(box_displacements) if len(box_displacements) > 1 else 0.0

            fps_proc = safe_div(1000, avg_latency)
            progreso = safe_div(frame_idx, total_frames) * 100
            info_color = (0, 255, 0) if hay_deteccion_actual else (200, 200, 200)
            cv.putText(annotated, f"FPS Display: {display_fps:.1f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(annotated, f"FPS Proc: {fps_proc:.1f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv.putText(annotated, f"Frame: {frame_idx}/{total_frames} ({progreso:.1f}%)", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 1)
            cv.putText(annotated, f"% Personas: {porcentaje:.1f}% | Conf: {avg_confidence:.2f}", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 255), 1)
            cv.putText(annotated, f"Backend: {DEVICE.upper()}", (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv.putText(annotated, f"Inf: {avg_inference:.1f}ms | Lat E2E: {avg_latency:.1f}ms", (10, 180), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv.putText(annotated, f"Inestabilidad: {indice_inestabilidad:.1f}%", (10, 210), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv.putText(annotated, f"Brusquedad(DSD): {dsd_running:.2f}", (10, 240), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            if paused:
                cv.putText(annotated, "PAUSADO", (10, 270), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv.imshow(window_name, annotated)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_event.set()
                break
            if key == ord(" "):
                paused = not paused

    print("[INFO] Finalizando...")
    stop_event.set()
    thread_capture.join()
    thread_process.join()
    cap.release()
    cv.destroyAllWindows()
    for _ in range(3):
        cv.waitKey(1)

    if processed_frames > 0:
        elapsed_total = time.time() - start_time
        final_display_fps = safe_div(display_frame_count, elapsed_total)
        avg_latency_ms = float(np.mean(latency_times)) if latency_times else 0.0
        final_proc_fps = safe_div(1000, avg_latency_ms)
        porc_persona_yolo = safe_div(total_person_frames, processed_frames) * 100
        avg_inference_ms = float(np.mean(inference_times)) if inference_times else 0.0
        avg_confidence = float(np.mean(person_confidences)) if person_confidences else 0.0
        indice_inestabilidad_final = safe_div(fallos_continuidad, display_frame_count) * 100
        dsd_final = np.std(box_displacements) if len(box_displacements) > 1 else 0.0
        detection_report = compute_detection_report(gt_by_frame, pred_by_frame, prediction_records, evaluated_frame_indices)
        df_iou_frames = pd.DataFrame(frame_iou_rows(gt_by_frame, pred_by_frame, evaluated_frame_indices, EVAL_IOU_THRESHOLD))

        df = pd.DataFrame(
            {
                "Run": [run_name],
                "Peso": [str(model_path)],
                "Video": [video_path.name],
                "Labels": [str(labels_path)],
                "Procesamiento": [f"Paralelo con Kalman ({DEVICE.upper()})"],
                "Resolucion": [f"{new_width}x{new_height}"],
                "Frames Omitidos (1 de cada)": [DETECT_EVERY],
                "Frames Totales (Video)": [total_frames],
                "Frames Procesados": [processed_frames],
                "Frames Mostrados": [display_frame_count],
                "FPS Promedio (Display)": [final_display_fps],
                "FPS Promedio (Inferencia)": [final_proc_fps],
                "Tiempo Total (s)": [elapsed_total],
                "% Personas Detectadas": [porc_persona_yolo],
                "Confianza Promedio": [avg_confidence],
                "Latencia Inferencia Promedio (ms)": [avg_inference_ms],
                "Latencia E2E Promedio (ms)": [avg_latency_ms],
                "Indice de Inestabilidad (%)": [indice_inestabilidad_final],
                "Brusquedad del Movimiento (DSD)": [dsd_final],
                "F1": [detection_report["F1"]],
                "Accuracy": [detection_report["Accuracy"]],
                "Precision": [detection_report["Precision"]],
                "Recall": [detection_report["Recall"]],
                "TP": [detection_report["TP"]],
                "TN": [detection_report["TN"]],
                "FP": [detection_report["FP"]],
                "FN": [detection_report["FN"]],
                "MAP5095": [detection_report["MAP5095"]],
                "MAP50": [detection_report["MAP50"]],
            }
        )

        write_xlsx_with_tabs(save_path, {"Resumen": df, "IoU_por_frame": df_iou_frames})
        print(f"[OK] Resultados guardados en: {save_path}")


if __name__ == "__main__":
    main()
