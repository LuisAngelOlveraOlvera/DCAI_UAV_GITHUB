import gc
import os
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import cv2 as cv
import numpy as np
import pandas as pd
from ultralytics import YOLO

from KalmanTracker import KalmanTracker
from config import DATASET_ROOT, EVALUATION_VIDEOS_DIR, EXPORTS_DIR


DETECT_EVERY = 3
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.4
MAX_DETECTIONS = 12
KALMAN_MAX_AGE = 10
KALMAN_MIN_HITS = 3
KALMAN_IOU_THRESHOLD = 0.3
EVAL_IOU_THRESHOLD = 0.30
MAP_IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)
COOLDOWN_SECONDS = 5
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg")
WEIGHTS_BASE_PATH = DATASET_ROOT / "runs" / "train"
OUTPUT_DIR = EXPORTS_DIR / "reportes_pesos_kalman_batch"


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
    rows.append(f'<row r="1">{"".join(excel_cell(header, 1, col_idx) for col_idx, header in enumerate(headers, start=1))}</row>')
    for row_idx, (_, row) in enumerate(df.iterrows(), start=2):
        rows.append(f'<row r="{row_idx}">{"".join(excel_cell(row[header], row_idx, col_idx) for col_idx, header in enumerate(headers, start=1))}</row>')
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
            if label_dir.is_dir() and any(label_dir.glob("frame_*.txt")) and str(label_dir) not in seen:
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


def average_precision(gt_by_frame, prediction_records, frame_indices, iou_threshold):
    total_gt = sum(len(gt_by_frame.get(frame_idx, [])) for frame_idx in frame_indices)
    if total_gt == 0:
        return 0.0
    matched = {frame_idx: set() for frame_idx in frame_indices}
    tp_values = []
    fp_values = []
    for pred in sorted(prediction_records, key=lambda item: item["score"], reverse=True):
        frame_idx = pred["frame_idx"]
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


def resolve_labels_for_video(video_path, labels_dirs):
    direct = video_path.parent / "labels"
    if direct.exists() and any(direct.glob("frame_*.txt")):
        return direct
    matches = [label_dir for label_dir in labels_dirs if label_dir.parent == video_path.parent]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No se encontro carpeta labels asociada a: {video_path}")


def choose_video(videos):
    print("\nVideos disponibles")
    print("------------------")
    for idx, video_path in enumerate(videos, start=1):
        print(f"{idx}. {video_path}")
    while True:
        raw = input("Selecciona un video por numero: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(videos):
                return videos[idx - 1]
        print("[WARN] Opcion invalida. Intenta de nuevo.")


def choose_weights(runs):
    print("\nPesos disponibles en runs\\train")
    print("-------------------------------")
    print("0. all")
    for idx, (run_name, _) in enumerate(runs, start=1):
        print(f"{idx}. {run_name}")
    while True:
        raw = input("Selecciona un numero, varios separados por coma, o 0/all: ").strip().lower()
        if raw in {"0", "all"}:
            return runs
        selected = []
        ok = True
        for piece in [part.strip() for part in raw.split(",") if part.strip()]:
            if not piece.isdigit():
                ok = False
                break
            idx = int(piece)
            if not 1 <= idx <= len(runs):
                ok = False
                break
            selected.append(runs[idx - 1])
        if ok and selected:
            seen = set()
            unique = []
            for item in selected:
                if item[0] not in seen:
                    seen.add(item[0])
                    unique.append(item)
            return unique
        print("[WARN] Seleccion invalida. Intenta de nuevo.")


def process_weight(run_name, weight_path, video_path, labels_path, output_dir):
    print(f"\n[INFO] Procesando {run_name}")
    model = YOLO(str(weight_path)).to(DEVICE)
    person_class = next((k for k, v in model.names.items() if v == "person"), 0)
    kalman_tracker = KalmanTracker(max_age=KALMAN_MAX_AGE, min_hits=KALMAN_MIN_HITS, iou_threshold=KALMAN_IOU_THRESHOLD)

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    start_time = time.time()
    processed_frames = 0
    frames_read = 0
    total_person_frames = 0
    inference_times = []
    confidences = []
    gt_by_frame = {}
    pred_by_frame = defaultdict(list)
    prediction_records = []
    evaluated_frame_indices = set()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_read += 1

        if frame_idx % DETECT_EVERY == 0:
            processed_frames += 1
            frame_h, frame_w = frame.shape[:2]
            inference_start = time.time()
            detections_for_kalman = []
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
            inference_times.append((time.time() - inference_start) * 1000)

            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                total_person_frames += 1
                xyxy_list = boxes.xyxy.cpu().numpy().tolist()
                scores = boxes.conf.cpu().numpy().tolist()
                confidences.extend(scores)
                for bbox, score in zip(xyxy_list, scores):
                    x1, y1, x2, y2 = bbox
                    detections_for_kalman.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, float(score), person_class])

            tracks = kalman_tracker.update(detections_for_kalman)
            frame_predictions = []
            for track in tracks:
                x, y, w, h, track_id, _, _ = track
                frame_predictions.append(
                    {
                        "frame_idx": frame_idx,
                        "bbox": [x - w / 2, y - h / 2, x + w / 2, y + h / 2],
                        "score": 1.0,
                        "track_id": int(track_id),
                    }
                )

            gt_by_frame[frame_idx] = load_ground_truth(labels_path, frame_idx, frame_w, frame_h)
            pred_by_frame[frame_idx] = [{"bbox": item["bbox"], "score": item["score"]} for item in frame_predictions]
            prediction_records.extend(frame_predictions)
            evaluated_frame_indices.add(frame_idx)

            if processed_frames % 50 == 0:
                progress = safe_div(frame_idx, total_frames) * 100
                print(f"[INFO] {run_name}: {processed_frames} inferencias ({progress:.1f}%)")
        else:
            kalman_tracker.update([])

        frame_idx += 1

    cap.release()
    elapsed_total = time.time() - start_time
    avg_inference_ms = float(np.mean(inference_times)) if inference_times else 0.0
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0
    detection_report = compute_detection_report(gt_by_frame, pred_by_frame, prediction_records, evaluated_frame_indices)

    summary = {
        "Modelo": run_name,
        "Peso": str(weight_path),
        "Video": video_path.name,
        "Labels": str(labels_path),
        "Resolucion": f"{video_width}x{video_height}",
        "Frames Omitidos (1 de cada)": DETECT_EVERY,
        "Frames Totales (Video)": total_frames,
        "Frames Leidos": frames_read,
        "Frames Procesados": processed_frames,
        "Tiempo Total (s)": elapsed_total,
        "FPS Promedio": safe_div(frames_read, elapsed_total),
        "FPS Inferencia": safe_div(1000, avg_inference_ms),
        "% Personas Detectadas": safe_div(total_person_frames, processed_frames) * 100,
        "Confianza Promedio": avg_confidence,
        "Latencia Inferencia Promedio (ms)": avg_inference_ms,
        "Umbral IoU Evaluacion": EVAL_IOU_THRESHOLD,
        **detection_report,
    }

    df_summary = pd.DataFrame([summary])
    df_iou = pd.DataFrame(frame_iou_rows(gt_by_frame, pred_by_frame, evaluated_frame_indices, EVAL_IOU_THRESHOLD))
    output_path = output_dir / f"{run_name}.xlsx"
    write_xlsx_with_tabs(output_path, {"Resumen": df_summary, "IoU_por_frame": df_iou})
    print(f"[OK] Excel generado: {output_path}")

    del model
    if torch is not None and DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    return summary


def main():
    runs = discover_weights()
    labels_dirs = discover_labels_directories()
    videos = discover_videos()

    if not runs:
        raise FileNotFoundError(f"No se encontraron pesos en: {WEIGHTS_BASE_PATH}")
    if not videos:
        raise FileNotFoundError("No se encontraron videos en las rutas exploradas del dataset.")
    if not labels_dirs:
        raise FileNotFoundError("No se encontraron carpetas labels con archivos frame_*.txt.")

    video_path = choose_video(videos)
    labels_path = resolve_labels_for_video(video_path, labels_dirs)
    selected_runs = choose_weights(runs)

    batch_dir = OUTPUT_DIR / video_path.stem
    batch_dir.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Configuracion seleccionada")
    print(f"Video: {video_path}")
    print(f"Labels: {labels_path}")
    print(f"Pesos seleccionados: {len(selected_runs)}")
    print(f"Salida: {batch_dir}")

    comparison_rows = []
    for idx, (run_name, weight_path) in enumerate(selected_runs, start=1):
        print(f"[INFO] Modelo {idx}/{len(selected_runs)}")
        comparison_rows.append(process_weight(run_name, weight_path, video_path, labels_path, batch_dir))
        if idx < len(selected_runs):
            print(f"[INFO] Enfriamiento de {COOLDOWN_SECONDS}s para liberar memoria...")
            time.sleep(COOLDOWN_SECONDS)

    final_path = batch_dir / "COMPARACION_FINAL.xlsx"
    write_xlsx_with_tabs(final_path, {"Comparacion": pd.DataFrame(comparison_rows)})
    print(f"\n[OK] Comparacion final generada: {final_path}")


if __name__ == "__main__":
    main()
