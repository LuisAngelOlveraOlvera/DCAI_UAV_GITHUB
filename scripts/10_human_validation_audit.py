"""
10 HUMAN VALIDATION AUDIT (judge evaluation)
-----------------------------------------------------------------------
Support human review of the judge model by:
1) preparing blind audit samples
2) collecting annotator decisions interactively
3) scoring consensus and agreement metrics
-----------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

import config
import config_analysis
import utils

cv2 = None


logger = utils.setup_logger(
    "Human_Validation_Audit",
    log_file=str(config.LOGS_DIR / "human_validation_audit.log"),
)

TRACKED_LABELS = ("ok", "missing_label", "bad_label", "poor_alignment")
HUMAN_LABELS = TRACKED_LABELS + ("ambiguous",)
DEFAULT_SAMPLE_SIZE = 10
DEFAULT_OK_AUDIT_SAMPLE = 10
DEFAULT_ANNOTATORS = 3
DEFAULT_OUT_DIR = config.EXPORTS_DIR / "human_validation_audit"
LISTA_ARCHIVOS_PATH = config.DATASET_ROOT / "scripts" / "lista_archivos.csv"
DEFAULT_PREPARE_STRATEGY = "all_judge_errors_plus_ok"
REVIEW_KEYMAP = {
    ord("1"): ("ok", "match", "high"),
    ord("2"): ("missing_label", "missing", "high"),
    ord("3"): ("bad_label", "bad", "high"),
    ord("4"): ("poor_alignment", "poor", "high"),
    ord("5"): ("ambiguous", "ambiguous", "medium"),
}


def print_table(df: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    if df.empty:
        print("[EMPTY]")
        return
    try:
        from tabulate import tabulate

        print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    except ImportError:
        print(df.to_string(index=False))


def fit_to_screen(img: np.ndarray, max_w: int = 1600, max_h: int = 950) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 2.25)
    if math.isclose(scale, 1.0):
        return img.copy(), 1.0
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    return resized, scale


def draw_gt_boxes(img: np.ndarray, boxes: Sequence[Sequence[int]], scale: float) -> np.ndarray:
    canvas = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = [int(v * scale) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return canvas


def draw_pred_boxes(
    img: np.ndarray, pred_boxes: Sequence[Dict[str, float | List[int]]], scale: float
) -> np.ndarray:
    canvas = img.copy()
    for pred in pred_boxes:
        x1, y1, x2, y2 = [int(v * scale) for v in pred["box"]]
        conf = float(pred["conf"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            canvas,
            f"{conf:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def render_review_frame(
    img: np.ndarray,
    boxes: Sequence[Sequence[int]],
    pred_boxes: Sequence[Dict[str, float | List[int]]],
    scale: float,
    row: pd.Series,
    current_pos: int,
    total_items: int,
    annotator_id: str,
) -> np.ndarray:
    overlay = draw_gt_boxes(img, boxes, scale)
    overlay = draw_pred_boxes(overlay, pred_boxes, scale)
    h, w = overlay.shape[:2]
    panel_h = 220
    panel = np.zeros((panel_h, w, 3), dtype=np.uint8)
    panel[:] = (28, 28, 28)

    current_label = str(row.get("human_label", "") or "").strip() or "pending"
    current_conf = str(row.get("annotator_confidence", "") or "").strip() or "-"
    judge_issue = str(row.get("auditor_label_issue", "") or "").strip() or "unknown"
    judge_score = row.get("judge_score", np.nan)
    judge_score_txt = f"{float(judge_score):.4f}" if pd.notna(judge_score) else "nan"
    judge_pred_count = int(row.get("judge_pred_count", 0) or 0)
    lines = [
        f"Annotator: {annotator_id} | Sample {current_pos}/{total_items} | ID: {row['sample_id']}",
        f"File: {row['file_name']} | GT boxes: {int(row.get('ann_count', 0) or 0)} | Current: {current_label} | Conf: {current_conf}",
        f"Judge issue: {judge_issue} | judge_score: {judge_score_txt} | stored_pred_count: {judge_pred_count} | live_pred_count: {len(pred_boxes)}",
        "Boxes: GT=green | Judge pred=red",
        "Keys: [1] ok  [2] missing_label  [3] bad_label  [4] poor_alignment  [5] ambiguous",
        "      [s] skip/next  [q or ESC] save and quit",
        "Human review should judge the existing annotation, not the auditor prediction.",
    ]

    for idx, text in enumerate(lines):
        cv2.putText(
            panel,
            text,
            (12, 28 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    return np.vstack([panel, overlay])


def label_to_binary(label: str) -> int:
    return int(str(label).strip() != "ok")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def derive_object_scale_bin(max_box_area_norm: float, ann_count: int) -> str:
    if ann_count <= 0 or pd.isna(max_box_area_norm):
        return "no_gt"
    if max_box_area_norm < 0.01:
        return "small"
    if max_box_area_norm < 0.10:
        return "medium"
    return "large"


def derive_judge_bin(score: float) -> str:
    if pd.isna(score):
        return "unknown"
    if score < config_analysis.JUDGE_RELAX:
        return "lt_0.35"
    if score < config_analysis.JUDGE_STRICT:
        return "0.35_0.65"
    return "ge_0.65"


def allocate_quota(group_sizes: pd.Series, total_quota: int) -> Dict[str, int]:
    if total_quota <= 0 or group_sizes.empty:
        return {str(idx): 0 for idx in group_sizes.index}

    total_available = int(group_sizes.sum())
    total_quota = min(total_quota, total_available)
    allocation = pd.Series(0, index=group_sizes.index, dtype=int)

    nonzero = group_sizes[group_sizes > 0]
    if nonzero.empty:
        return allocation.to_dict()

    if total_quota >= len(nonzero):
        allocation.loc[nonzero.index] = 1
        remaining = total_quota - len(nonzero)
    else:
        ordered = nonzero.sort_values(ascending=False)
        allocation.loc[ordered.index[:total_quota]] = 1
        remaining = 0

    if remaining > 0:
        capacity = group_sizes - allocation
        positive_capacity = capacity[capacity > 0]
        if not positive_capacity.empty:
            weights = positive_capacity / positive_capacity.sum()
            raw_extra = weights * remaining
            floor_extra = np.floor(raw_extra).astype(int)
            allocation.loc[floor_extra.index] += floor_extra
            used = int(floor_extra.sum())
            leftover = remaining - used
            if leftover > 0:
                frac = (raw_extra - floor_extra).sort_values(ascending=False)
                for idx in frac.index:
                    if leftover <= 0:
                        break
                    if allocation.loc[idx] < group_sizes.loc[idx]:
                        allocation.loc[idx] += 1
                        leftover -= 1

    allocation = allocation.clip(upper=group_sizes)
    return allocation.astype(int).to_dict()


def cohen_kappa_score_manual(labels_a: Sequence[str], labels_b: Sequence[str], classes: Sequence[str]) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        return float("nan")

    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum(
        safe_div(counts_a.get(cls, 0), n) * safe_div(counts_b.get(cls, 0), n)
        for cls in classes
    )
    if math.isclose(1.0 - expected, 0.0):
        return 1.0
    return (observed - expected) / (1.0 - expected)


def fleiss_kappa_manual(ratings: List[List[str]], classes: Sequence[str]) -> float:
    if not ratings:
        return float("nan")

    n_raters = len(ratings[0])
    if n_raters < 2:
        return float("nan")
    if any(len(row) != n_raters for row in ratings):
        return float("nan")

    n_items = len(ratings)
    counts = np.zeros((n_items, len(classes)), dtype=float)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    for i, row in enumerate(ratings):
        row_counter = Counter(row)
        for cls, cnt in row_counter.items():
            if cls in class_to_idx:
                counts[i, class_to_idx[cls]] = cnt

    p_j = counts.sum(axis=0) / (n_items * n_raters)
    p_i = ((counts * counts).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_e = (p_j * p_j).sum()
    if math.isclose(1.0 - p_e, 0.0):
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)


def binary_classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    if not y_true:
        return {
            "n": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "accuracy": float("nan"),
        }

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    accuracy = safe_div(tp + tn, len(y_true))
    return {
        "n": len(y_true),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def build_base_dataframe() -> pd.DataFrame:
    if not config_analysis.ANALYSIS_DB_PATH.exists():
        raise FileNotFoundError(f"No existe DB de analisis: {config_analysis.ANALYSIS_DB_PATH}")
    if not config.DB_PATH.exists():
        raise FileNotFoundError(f"No existe DB maestra: {config.DB_PATH}")

    conn_analysis = sqlite3.connect(config_analysis.ANALYSIS_DB_PATH)
    conn_master = sqlite3.connect(config.DB_PATH)
    try:
        df_analysis = pd.read_sql(
            """
            SELECT
                id AS analysis_id,
                file_name,
                relative_path,
                source,
                judge_score,
                label_issue AS auditor_label_issue,
                judge_has_pred,
                judge_pred_count,
                judge_max_iou,
                dedup_cluster,
                is_dedup_keep,
                processed
            FROM analysis_images
            """,
            conn_analysis,
        )

        df_images = pd.read_sql(
            """
            SELECT
                id AS image_id,
                file_name,
                relative_path,
                width,
                height,
                label_issue AS master_label_issue
            FROM images
            """,
            conn_master,
        )

        df_ann = pd.read_sql(
            """
            SELECT
                image_id,
                COUNT(*) AS ann_count,
                MIN(width * height) AS min_box_area_norm,
                MAX(width * height) AS max_box_area_norm
            FROM annotations
            GROUP BY image_id
            """,
            conn_master,
        )
    finally:
        conn_analysis.close()
        conn_master.close()

    df = df_analysis.merge(
        df_images[["image_id", "relative_path", "width", "height", "master_label_issue"]],
        on="relative_path",
        how="left",
    )
    df = df.merge(df_ann, on="image_id", how="left")

    df["ann_count"] = df["ann_count"].fillna(0).astype(int)
    df["max_box_area_norm"] = df["max_box_area_norm"].astype(float)
    df["min_box_area_norm"] = df["min_box_area_norm"].astype(float)
    df["auditor_label_issue"] = df["auditor_label_issue"].fillna("unknown")
    df["master_label_issue"] = df["master_label_issue"].fillna("unknown")

    if "is_dedup_keep" in df.columns:
        dedup_mask = df["is_dedup_keep"].isna() | (df["is_dedup_keep"] == 1)
        df = df[dedup_mask].copy()

    if "processed" in df.columns:
        df = df[df["processed"] != 2].copy()

    df = df[df["auditor_label_issue"].isin(TRACKED_LABELS)].copy()
    df["image_path_abs"] = df["relative_path"].astype(str)
    df["object_scale_bin"] = df.apply(
        lambda row: derive_object_scale_bin(row["max_box_area_norm"], int(row["ann_count"])),
        axis=1,
    )
    df["judge_conf_bin"] = df["judge_score"].map(derive_judge_bin)
    df["binary_auditor_error"] = df["auditor_label_issue"].map(label_to_binary)

    if LISTA_ARCHIVOS_PATH.exists():
        df_dist = pd.read_csv(LISTA_ARCHIVOS_PATH)
        df_dist.columns = df_dist.columns.str.strip()
        if {"Nombre", "DISTANCIA"}.issubset(df_dist.columns):
            df_dist = df_dist.rename(columns={"Nombre": "file_name", "DISTANCIA": "distance_m"})
            df_dist["distance_m"] = pd.to_numeric(df_dist["distance_m"], errors="coerce")
            df = df.merge(df_dist[["file_name", "distance_m"]], on="file_name", how="left")
        else:
            df["distance_m"] = np.nan
    else:
        df["distance_m"] = np.nan

    df["distance_bin"] = df["distance_m"].apply(
        lambda v: f"{int(v)}m" if pd.notna(v) else "unknown"
    )
    return df


def load_gt_boxes_for_paths(relative_paths: Iterable[str]) -> Dict[str, List[List[int]]]:
    clean_paths = [str(p) for p in relative_paths if str(p).strip()]
    if not clean_paths:
        return {}

    boxes_by_path: Dict[str, List[List[int]]] = {path: [] for path in clean_paths}
    conn = sqlite3.connect(config.DB_PATH)
    try:
        chunk_size = 800
        for start in range(0, len(clean_paths), chunk_size):
            chunk = clean_paths[start:start + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            rows = conn.execute(
                f"""
                SELECT
                    i.relative_path,
                    i.width,
                    i.height,
                    a.x_center,
                    a.y_center,
                    a.width,
                    a.height
                FROM images i
                LEFT JOIN annotations a ON a.image_id = i.id
                WHERE i.relative_path IN ({placeholders})
                """,
                chunk,
            ).fetchall()

            for rel_path, img_w, img_h, xc, yc, bw, bh in rows:
                boxes_by_path.setdefault(rel_path, [])
                if xc is None:
                    continue
                box = utils.xywh_norm_to_xyxy_abs([xc, yc, bw, bh], img_w, img_h)
                boxes_by_path[rel_path].append([int(v) for v in box])
    finally:
        conn.close()

    return boxes_by_path


def load_judge_model():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics es requerido para mostrar cajas del modelo juez.") from exc

    model_path = Path(config_analysis.MODEL_JUDGE_PATH)
    if not model_path.exists():
        model_path = config.DATASET_ROOT / str(config_analysis.MODEL_JUDGE_PATH)
    if not model_path.exists():
        model_path = Path("yolo11x.pt")

    return YOLO(str(model_path))


def predict_judge_boxes(model, img_path: Path) -> List[Dict[str, float | List[int]]]:
    results = model.predict(
        source=str(img_path),
        classes=[0],
        conf=config.CONF_THRESHOLD,
        verbose=False,
    )

    pred_boxes: List[Dict[str, float | List[int]]] = []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return pred_boxes

    confs = boxes.conf.detach().cpu().tolist()
    xyxy = boxes.xyxy.detach().cpu().tolist()
    for conf, box in zip(confs, xyxy):
        pred_boxes.append({"box": [int(v) for v in box], "conf": float(conf)})
    return pred_boxes


def stratified_human_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    issue_counts = df["auditor_label_issue"].value_counts()
    active_issues = [label for label in TRACKED_LABELS if issue_counts.get(label, 0) > 0]
    if not active_issues:
        raise ValueError("No hay muestras elegibles para validacion humana.")

    base_quota = sample_size // len(active_issues)
    issue_quota = {label: min(base_quota, issue_counts.get(label, 0)) for label in active_issues}
    remaining = sample_size - sum(issue_quota.values())
    leftovers = {label: int(issue_counts.get(label, 0) - issue_quota[label]) for label in active_issues}
    for label in sorted(leftovers, key=leftovers.get, reverse=True):
        if remaining <= 0:
            break
        extra = min(leftovers[label], remaining)
        issue_quota[label] += extra
        remaining -= extra

    sampled_parts = []
    strata_cols = ["source", "object_scale_bin", "judge_conf_bin", "distance_bin"]
    for label in active_issues:
        quota = issue_quota[label]
        df_issue = df[df["auditor_label_issue"] == label].copy()
        if quota <= 0 or df_issue.empty:
            continue

        df_issue["stratum"] = df_issue[strata_cols].fillna("unknown").astype(str).agg("|".join, axis=1)
        allocation = allocate_quota(df_issue["stratum"].value_counts(), quota)

        local_rows = []
        for stratum, stratum_quota in allocation.items():
            if stratum_quota <= 0:
                continue
            df_group = df_issue[df_issue["stratum"] == stratum].copy()
            order = rng.permutation(len(df_group))
            local_rows.append(df_group.iloc[order[:stratum_quota]])

        if local_rows:
            sampled_issue = pd.concat(local_rows, axis=0)
        else:
            sampled_issue = df_issue.sample(n=min(quota, len(df_issue)), random_state=seed)

        sampled_parts.append(sampled_issue)

    if not sampled_parts:
        raise ValueError("No se pudo construir la muestra estratificada.")

    sampled = pd.concat(sampled_parts, axis=0).drop_duplicates(subset=["analysis_id"]).copy()
    if len(sampled) < min(sample_size, len(df)):
        needed = min(sample_size, len(df)) - len(sampled)
        remaining_df = df[~df["analysis_id"].isin(sampled["analysis_id"])].copy()
        if needed > 0 and not remaining_df.empty:
            sampled = pd.concat(
                [sampled, remaining_df.sample(n=min(needed, len(remaining_df)), random_state=seed)],
                axis=0,
            )

    sampled = sampled.sort_values(["auditor_label_issue", "source", "file_name"]).reset_index(drop=True)
    sampled["sample_id"] = [f"HV_{idx:04d}" for idx in range(1, len(sampled) + 1)]
    return sampled


def stratified_subset_by_columns(
    df_in: pd.DataFrame,
    target_n: int,
    seed: int,
    strata_cols: Sequence[str],
) -> pd.DataFrame:
    if df_in.empty or target_n <= 0:
        return df_in.iloc[0:0].copy()

    target_n = min(target_n, len(df_in))
    rng = np.random.default_rng(seed)
    df_work = df_in.copy()
    df_work["stratum"] = df_work[list(strata_cols)].fillna("unknown").astype(str).agg("|".join, axis=1)
    allocation = allocate_quota(df_work["stratum"].value_counts(), target_n)

    sampled_parts = []
    for stratum, quota in allocation.items():
        if quota <= 0:
            continue
        df_group = df_work[df_work["stratum"] == stratum].copy()
        order = rng.permutation(len(df_group))
        sampled_parts.append(df_group.iloc[order[:quota]])

    sampled = pd.concat(sampled_parts, axis=0).drop_duplicates(subset=["analysis_id"])
    if len(sampled) < target_n:
        remaining = df_work[~df_work["analysis_id"].isin(sampled["analysis_id"])].copy()
        needed = min(target_n - len(sampled), len(remaining))
        if needed > 0:
            sampled = pd.concat(
                [sampled, remaining.sample(n=needed, random_state=seed)],
                axis=0,
            )

    return sampled.drop(columns=["stratum"], errors="ignore").copy()


def required_human_audit_sample(df: pd.DataFrame, ok_sample_size: int, seed: int) -> pd.DataFrame:
    df_errors = df[df["auditor_label_issue"] != "ok"].copy()
    df_ok = df[df["auditor_label_issue"] == "ok"].copy()

    ok_sample = stratified_subset_by_columns(
        df_ok,
        target_n=ok_sample_size,
        seed=seed,
        strata_cols=["source", "object_scale_bin", "judge_conf_bin", "distance_bin"],
    )

    sampled = pd.concat([df_errors, ok_sample], axis=0).drop_duplicates(subset=["analysis_id"])
    sampled = sampled.sort_values(["auditor_label_issue", "source", "file_name"]).reset_index(drop=True)
    sampled["sample_id"] = [f"HV_{idx:04d}" for idx in range(1, len(sampled) + 1)]
    return sampled


def export_prepare_outputs(sample_df: pd.DataFrame, out_dir: Path, annotators: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    master_cols = [
        "sample_id",
        "analysis_id",
        "image_id",
        "file_name",
        "relative_path",
        "image_path_abs",
        "source",
        "distance_m",
        "distance_bin",
        "width",
        "height",
        "ann_count",
        "object_scale_bin",
        "judge_score",
        "judge_conf_bin",
        "judge_pred_count",
        "judge_max_iou",
        "auditor_label_issue",
        "binary_auditor_error",
    ]
    master_path = out_dir / "human_validation_sample_master.csv"
    sample_df[master_cols].to_csv(master_path, index=False)

    blind_cols = [
        "sample_id",
        "file_name",
        "relative_path",
        "image_path_abs",
        "width",
        "height",
        "distance_m",
    ]
    blind_df = sample_df[blind_cols].copy()
    blind_df["human_label"] = ""
    blind_df["bbox_quality"] = ""
    blind_df["annotator_confidence"] = ""
    blind_df["notes"] = ""

    blind_path = out_dir / "human_validation_blind_template.csv"
    blind_df.to_csv(blind_path, index=False)

    for idx in range(1, annotators + 1):
        annotator_df = blind_df.copy()
        annotator_df.insert(1, "annotator_id", f"A{idx}")
        annotator_path = out_dir / f"human_validation_annotator_A{idx}.csv"
        annotator_df.to_csv(annotator_path, index=False)

    summary = (
        sample_df.groupby(["auditor_label_issue", "source", "object_scale_bin"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["auditor_label_issue", "source", "object_scale_bin"])
    )
    summary_path = out_dir / "human_validation_sample_summary.csv"
    summary.to_csv(summary_path, index=False)
    print_table(summary, "HUMAN VALIDATION SAMPLE SUMMARY")

    print(f"\nSaved: {master_path}")
    print(f"Saved: {blind_path}")
    for idx in range(1, annotators + 1):
        print(f"Saved: {out_dir / f'human_validation_annotator_A{idx}.csv'}")
    print(f"Saved: {summary_path}")


def prepare_mode(args: argparse.Namespace) -> None:
    df = build_base_dataframe()
    if args.strategy == "all_judge_errors_plus_ok":
        sample_df = required_human_audit_sample(
            df,
            ok_sample_size=args.ok_sample_size,
            seed=args.seed,
        )
        n_errors = int((sample_df["auditor_label_issue"] != "ok").sum())
        n_ok = int((sample_df["auditor_label_issue"] == "ok").sum())
        print(
            f"[INFO] prepare strategy={args.strategy} | errores del juez={n_errors} | "
            f"ok_sample={n_ok} | total={len(sample_df)}"
        )
    else:
        sample_df = stratified_human_sample(df, sample_size=args.sample_size, seed=args.seed)
        print(
            f"[INFO] prepare strategy={args.strategy} | sample_size={len(sample_df)}"
        )
    export_prepare_outputs(sample_df, out_dir=args.out_dir, annotators=args.annotators)


def review_mode(args: argparse.Namespace) -> None:
    global cv2
    if cv2 is None:
        try:
            import cv2 as cv2_module
        except ImportError as exc:
            raise ImportError("OpenCV (cv2) es requerido para review mode.") from exc
        cv2 = cv2_module

    annotator_path = args.out_dir / f"human_validation_annotator_{args.annotator_id}.csv"
    if not annotator_path.exists():
        raise FileNotFoundError(f"No existe archivo de anotador: {annotator_path}")
    master_path = args.out_dir / "human_validation_sample_master.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"No existe sample master: {master_path}")

    df = pd.read_csv(annotator_path)
    master_df = pd.read_csv(master_path)
    master_keep = [
        "sample_id",
        "ann_count",
        "auditor_label_issue",
        "judge_score",
        "judge_pred_count",
        "judge_max_iou",
    ]
    df = df.drop(columns=[col for col in master_keep if col != "sample_id" and col in df.columns], errors="ignore")
    df = df.merge(master_df[master_keep], on="sample_id", how="left")
    for col in ["human_label", "bbox_quality", "annotator_confidence", "notes", "reviewed_at"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    if "annotator_id" not in df.columns:
        df.insert(1, "annotator_id", args.annotator_id)

    pending_mask = df["human_label"].astype(str).str.strip().eq("")
    if args.overwrite:
        indices = df.index.tolist()
    else:
        indices = df.index[pending_mask].tolist()

    if not indices:
        print(f"[INFO] No hay muestras pendientes para {args.annotator_id}.")
        return

    gt_boxes_map = load_gt_boxes_for_paths(df.loc[indices, "relative_path"].tolist())
    judge_model = load_judge_model()
    window_name = f"HumanReview-{args.annotator_id}"

    for pos, idx in enumerate(indices, start=1):
        row = df.loc[idx].copy()
        raw_path = str(row["image_path_abs"]).strip()
        if raw_path:
            img_path = Path(raw_path)
            if not img_path.is_absolute():
                img_path = config.DATASET_ROOT / img_path
        else:
            img_path = config.DATASET_ROOT / str(row["relative_path"])
        if not img_path.exists():
            logger.warning(f"Imagen no encontrada: {img_path}")
            continue

        img_raw = cv2.imread(str(img_path))
        if img_raw is None:
            logger.warning(f"No se pudo leer imagen: {img_path}")
            continue

        img_fit, scale = fit_to_screen(img_raw)
        gt_boxes = gt_boxes_map.get(str(row["relative_path"]), [])
        pred_boxes = predict_judge_boxes(judge_model, img_path)

        while True:
            frame = render_review_frame(
                img_fit,
                gt_boxes,
                pred_boxes,
                scale,
                row,
                current_pos=pos,
                total_items=len(indices),
                annotator_id=args.annotator_id,
            )
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(0) & 0xFF

            if key in REVIEW_KEYMAP:
                label, bbox_quality, confidence = REVIEW_KEYMAP[key]
                df.at[idx, "human_label"] = label
                df.at[idx, "bbox_quality"] = bbox_quality
                df.at[idx, "annotator_confidence"] = confidence
                df.at[idx, "annotator_id"] = args.annotator_id
                df.at[idx, "reviewed_at"] = pd.Timestamp.now().isoformat()
                df.to_csv(annotator_path, index=False)
                print(f"[SAVED] {row['sample_id']} -> {label}")
                break

            if key in (ord("s"), ord("S"), 32):
                break

            if key in (ord("q"), ord("Q"), 27):
                df.to_csv(annotator_path, index=False)
                cv2.destroyAllWindows()
                print(f"[STOP] Revisi\xC3\xB3n pausada. Archivo guardado: {annotator_path}")
                return

        row = df.loc[idx].copy()

    df.to_csv(annotator_path, index=False)
    cv2.destroyAllWindows()
    print(f"[DONE] Revisi\xC3\xB3n completada para {args.annotator_id}: {annotator_path}")


def load_annotation_files(out_dir: Path, annotations_glob: str) -> pd.DataFrame:
    paths = sorted(out_dir.glob(annotations_glob))
    if not paths:
        raise FileNotFoundError(
            f"No se encontraron archivos de anotacion con patron: {out_dir / annotations_glob}"
        )

    parts = []
    for path in paths:
        df = pd.read_csv(path)
        df["annotation_file"] = path.name
        parts.append(df)
    return pd.concat(parts, axis=0, ignore_index=True)


def build_consensus(df_annotations: pd.DataFrame, expected_annotators: int) -> pd.DataFrame:
    rows = []
    for sample_id, df_sample in df_annotations.groupby("sample_id"):
        labels = [
            str(v).strip()
            for v in df_sample["human_label"].fillna("").tolist()
            if str(v).strip() != ""
        ]
        label_counter = Counter(labels)
        completed = len(labels)
        consensus_label = "pending"
        consensus_strength = 0.0

        if completed > 0:
            top_label, top_count = label_counter.most_common(1)[0]
            consensus_strength = safe_div(top_count, completed)
            if top_count > completed / 2:
                consensus_label = top_label
            elif completed >= 2:
                consensus_label = "ambiguous"

        rows.append(
            {
                "sample_id": sample_id,
                "n_completed": completed,
                "expected_annotators": expected_annotators,
                "consensus_label": consensus_label,
                "consensus_strength": consensus_strength,
                "label_counts": "; ".join(f"{k}:{v}" for k, v in sorted(label_counter.items())),
            }
        )

    return pd.DataFrame(rows)


def pairwise_kappa_table(df_annotations: pd.DataFrame) -> pd.DataFrame:
    annotators = sorted(df_annotations["annotator_id"].dropna().astype(str).unique().tolist())
    rows = []
    for a_id, b_id in combinations(annotators, 2):
        df_a = df_annotations[df_annotations["annotator_id"].astype(str) == a_id][["sample_id", "human_label"]]
        df_b = df_annotations[df_annotations["annotator_id"].astype(str) == b_id][["sample_id", "human_label"]]
        merged = df_a.merge(df_b, on="sample_id", suffixes=("_a", "_b"))
        merged = merged[
            merged["human_label_a"].fillna("").astype(str).str.strip().ne("")
            & merged["human_label_b"].fillna("").astype(str).str.strip().ne("")
        ].copy()

        labels_a = merged["human_label_a"].astype(str).tolist()
        labels_b = merged["human_label_b"].astype(str).tolist()
        kappa_multiclass = cohen_kappa_score_manual(labels_a, labels_b, HUMAN_LABELS)
        binary_a = [label_to_binary(v) for v in labels_a]
        binary_b = [label_to_binary(v) for v in labels_b]
        kappa_binary = cohen_kappa_score_manual(
            [str(v) for v in binary_a], [str(v) for v in binary_b], ("0", "1")
        )
        rows.append(
            {
                "annotator_a": a_id,
                "annotator_b": b_id,
                "n_overlap": len(merged),
                "cohen_kappa_multiclass": kappa_multiclass,
                "cohen_kappa_binary": kappa_binary,
            }
        )

    return pd.DataFrame(rows)


def fleiss_ready_rows(df_annotations: pd.DataFrame, expected_annotators: int) -> List[List[str]]:
    ratings = []
    for _sample_id, df_sample in df_annotations.groupby("sample_id"):
        labels = [
            str(v).strip()
            for v in df_sample["human_label"].fillna("").tolist()
            if str(v).strip() != ""
        ]
        if len(labels) == expected_annotators:
            ratings.append(labels)
    return ratings


def auditor_vs_human_tables(master_df: pd.DataFrame, consensus_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = master_df.merge(consensus_df, on="sample_id", how="left")
    valid = merged[merged["consensus_label"].isin(TRACKED_LABELS)].copy()

    y_true = valid["consensus_label"].map(label_to_binary).tolist()
    y_pred = valid["auditor_label_issue"].map(label_to_binary).tolist()
    overall = pd.DataFrame([binary_classification_metrics(y_true, y_pred)])

    class_rows = []
    for label in TRACKED_LABELS:
        y_true_cls = [int(v == label) for v in valid["consensus_label"].tolist()]
        y_pred_cls = [int(v == label) for v in valid["auditor_label_issue"].tolist()]
        metrics_cls = binary_classification_metrics(y_true_cls, y_pred_cls)
        metrics_cls["label"] = label
        class_rows.append(metrics_cls)
    per_class = pd.DataFrame(class_rows)[
        ["label", "n", "tp", "fp", "tn", "fn", "precision", "recall", "f1", "accuracy"]
    ]

    breakdown_rows = []
    for column in ["source", "object_scale_bin", "distance_bin", "judge_conf_bin"]:
        if column not in valid.columns:
            continue
        for group_value, df_group in valid.groupby(column, dropna=False):
            metrics_group = binary_classification_metrics(
                df_group["consensus_label"].map(label_to_binary).tolist(),
                df_group["auditor_label_issue"].map(label_to_binary).tolist(),
            )
            breakdown_rows.append(
                {
                    "breakdown": column,
                    "group": group_value,
                    **metrics_group,
                }
            )
    breakdown = pd.DataFrame(breakdown_rows)

    if not overall.empty:
        overall.insert(0, "scope", "auditor_vs_human_consensus_binary")
    return overall, per_class, breakdown


def score_mode(args: argparse.Namespace) -> None:
    out_dir = args.out_dir
    master_path = out_dir / "human_validation_sample_master.csv"
    if not master_path.exists():
        raise FileNotFoundError(f"No existe sample master: {master_path}")

    master_df = pd.read_csv(master_path)
    annotations_df = load_annotation_files(out_dir, args.annotations_glob)

    required_cols = {"sample_id", "annotator_id", "human_label"}
    missing = required_cols - set(annotations_df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en anotaciones humanas: {sorted(missing)}")

    annotations_df["human_label"] = (
        annotations_df["human_label"].fillna("").astype(str).str.strip().str.lower()
    )
    invalid_labels = sorted(
        set(v for v in annotations_df["human_label"].unique().tolist() if v and v not in HUMAN_LABELS)
    )
    if invalid_labels:
        raise ValueError(f"Etiquetas humanas no validas: {invalid_labels}")

    consensus_df = build_consensus(annotations_df, expected_annotators=args.expected_annotators)
    pairwise_df = pairwise_kappa_table(annotations_df)
    fleiss_rows = fleiss_ready_rows(annotations_df, expected_annotators=args.expected_annotators)
    fleiss_value = fleiss_kappa_manual(fleiss_rows, HUMAN_LABELS)

    overall_df, per_class_df, breakdown_df = auditor_vs_human_tables(master_df, consensus_df)
    overall_df["fleiss_kappa_multiclass"] = fleiss_value

    consensus_path = out_dir / "human_validation_consensus.csv"
    pairwise_path = out_dir / "human_validation_pairwise_kappa.csv"
    overall_path = out_dir / "human_validation_metrics_summary.csv"
    per_class_path = out_dir / "human_validation_metrics_per_class.csv"
    breakdown_path = out_dir / "human_validation_metrics_breakdown.csv"

    consensus_df.to_csv(consensus_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)
    overall_df.to_csv(overall_path, index=False)
    per_class_df.to_csv(per_class_path, index=False)
    breakdown_df.to_csv(breakdown_path, index=False)

    print_table(consensus_df.head(15), "CONSENSUS PREVIEW")
    print_table(overall_df, "AUDITOR VS HUMAN CONSENSUS")
    print_table(per_class_df, "PER-CLASS METRICS")
    print_table(pairwise_df, "PAIRWISE COHEN KAPPA")
    if not breakdown_df.empty:
        print_table(breakdown_df, "BREAKDOWN METRICS")

    print(f"\nSaved: {consensus_path}")
    print(f"Saved: {pairwise_path}")
    print(f"Saved: {overall_path}")
    print(f"Saved: {per_class_path}")
    print(f"Saved: {breakdown_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Human validation audit for the judge model."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    parser_prepare = subparsers.add_parser("prepare", help="Build human validation sample.")
    parser_prepare.add_argument(
        "--strategy",
        choices=["all_judge_errors_plus_ok", "mixed_stratified"],
        default=DEFAULT_PREPARE_STRATEGY,
    )
    parser_prepare.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser_prepare.add_argument("--ok-sample-size", type=int, default=DEFAULT_OK_AUDIT_SAMPLE)
    parser_prepare.add_argument("--annotators", type=int, default=DEFAULT_ANNOTATORS)
    parser_prepare.add_argument("--seed", type=int, default=config.SEED)
    parser_prepare.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser_prepare.set_defaults(func=prepare_mode)

    parser_review = subparsers.add_parser("review", help="Interactive human review, image by image.")
    parser_review.add_argument("--annotator-id", default="A1")
    parser_review.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser_review.add_argument("--overwrite", action="store_true")
    parser_review.set_defaults(func=review_mode)

    parser_score = subparsers.add_parser("score", help="Score completed human annotations.")
    parser_score.add_argument("--annotations-glob", default="human_validation_annotator_A*.csv")
    parser_score.add_argument("--expected-annotators", type=int, default=DEFAULT_ANNOTATORS)
    parser_score.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser_score.set_defaults(func=score_mode)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
