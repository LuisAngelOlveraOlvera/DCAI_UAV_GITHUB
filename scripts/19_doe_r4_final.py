"""
19 R4 FINAL SELECTION (post-DoE ranking)
-----------------------------------------------------------------------
Select the final winner after R0-R3 by:
1) consolidating finalist metrics
2) applying the R4 decision logic
3) exporting the final ranking and supporting outputs
-----------------------------------------------------------------------
"""

from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.colors import LinearSegmentedColormap
from ultralytics import YOLO

import config
import utils


logger = utils.setup_logger(
    "DOE_R4_Final",
    log_file=str(config.LOGS_DIR / "doe_r4_final.log"),
)

DATASET_COL = "Dataset"
MODEL_ID_COL = "ModelID"
SCENARIO_COL = "scenario_canonical"
METRICS = ["mAP50-95", "mAP50", "Precision", "Recall"]
METRIC_LABELS = {
    "mAP50-95": "mAP@0.5:0.95",
    "mAP50": "mAP@0.5",
    "Precision": "Precision",
    "Recall": "Recall",
}

R4_NAME = "R4_FINAL"
OUT_DIR = config.EXPORTS_DIR / R4_NAME
FIGS_DIR = OUT_DIR / f"{R4_NAME}_figs"
R4_RUNS_DIR = config.DATASET_ROOT / "runs" / "RONDA_4"
RAW_EVAL_CSV = OUT_DIR / f"{R4_NAME}_raw_eval_per_seed_dataset.csv"
TMP_YAML_DIR = OUT_DIR / "_tmp_eval_yaml"

# Fill with canonical labels from R0-R3, e.g.: "B_Raw_0", "H_pH_JR_15", "P_pH_JR_15"
R4_FINALISTS: List[str] = [
    "Z_COCO",
    "B_Raw_0",
    "H_pH_JS_0",
    "H_pH_JS_40",
    "P_Raw_0",
]

ALPHA = 0.50
WEIGHTS: Dict[str, float] = {
    "mAP50-95": 0.50,
    "mAP50": 0.20,
    "Precision": 0.15,
    "Recall": 0.15,
}

GATING_MODE = "anchor"  # "anchor" or "none"
GATING_ANCHOR = "P_Raw_0"
GATING_FACTOR = 1.00

HEATMAP_COLORS = ["#ffffff", "#fff3b0", "#ffb347", "#d7301f"]
ABS_CMAP = LinearSegmentedColormap.from_list("white_yellow_orange_red_abs", HEATMAP_COLORS)
DELTA_CMAP = LinearSegmentedColormap.from_list("white_yellow_orange_red_delta", HEATMAP_COLORS)
DELTA_SCALE = 100.0
PERCENT_SCALE = 100.0
EVAL_BATCH = 8
EVAL_IMGSZ = 640
EVAL_USE_CACHE = True

# Reused test domains for robust cross-domain evaluation.
TEST_DATASETS: Dict[str, Path] = {
    "VISDRONE": Path(r"EVALUATION\VISDRONE"),
    "COCO_TEST": Path(r"EVALUATION\COCO_TEST"),
    "MANIPAL_UAV": Path(r"EVALUATION\MANIPAL_UAV"),
    "IRINA": Path(r"EVALUATION\IRINA"),
    "NTUT": Path(r"EVALUATION\NTUT"),
    "DOMINIO_DRON": Path(r"EVALUATION\UAQ_MSUAV_TEST"),
}


EXPECTED_R4_SEEDS: Set[int] = {0, 7, 42, 123, 999}
MODEL_ID_ALIAS = {
    "N0": "N14",
}
N_TO_CANONICAL = {
    "N1": "Z_COCO",
    "N2": "B_Raw_0",
    "N7": "H_pH_JS_0",
    "N10": "H_pH_JS_40",
    "N11": "P_Raw_0",
}


CANONICAL_TO_N = {v: k for k, v in N_TO_CANONICAL.items()}


def normalize_scenario(raw_name: str) -> str:
    raw = str(raw_name).strip()
    if raw in R4_FINALISTS:
        return raw

    n_id = infer_model_id_from_text(raw)
    if n_id is not None:
        mapped = N_TO_CANONICAL.get(n_id)
        if mapped:
            return mapped

    for canonical in R4_FINALISTS:
        if canonical in raw:
            return canonical
    return raw


def infer_seed_from_text(value: str) -> int:
    txt = str(value)
    m = re.search(r"_seed(\d+)", txt)
    if m:
        return int(m.group(1))
    return 0


def normalize_model_id(model_id: str) -> str:
    mid = str(model_id).strip()
    return MODEL_ID_ALIAS.get(mid, mid)


def infer_model_id_from_text(value: str) -> str | None:
    txt = str(value).strip()
    m = re.match(r"^(N\d+)", txt)
    if not m:
        return None
    return normalize_model_id(m.group(1))


def clean_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def create_test_yaml(dataset_dir: Path, yaml_path: Path) -> Path:
    data = {
        "path": str(dataset_dir.absolute()),
        "train": "images",
        "val": "images",
        "test": "images",
        "names": {0: "person"},
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return yaml_path


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def extract_metrics(metrics: object) -> Dict[str, float]:
    box = getattr(metrics, "box", None)
    if box is None:
        return {m: float("nan") for m in METRICS}
    return {
        "mAP50-95": safe_float(getattr(box, "map", float("nan"))),
        "mAP50": safe_float(getattr(box, "map50", float("nan"))),
        "Precision": safe_float(getattr(box, "mp", float("nan"))),
        "Recall": safe_float(getattr(box, "mr", float("nan"))),
    }


def discover_r4_weights() -> pd.DataFrame:
    if not R4_RUNS_DIR.exists():
        raise FileNotFoundError(f"No existe el directorio R4: {R4_RUNS_DIR}")

    rows = []
    for best_pt in sorted(R4_RUNS_DIR.glob("*/weights/best.pt")):
        run_dir = best_pt.parent.parent
        run_name = run_dir.name
        model_id = infer_model_id_from_text(run_name)
        scenario = normalize_scenario(run_name)
        if model_id is None:
            model_id = CANONICAL_TO_N.get(scenario, "NA")
        seed = infer_seed_from_text(run_name)
        rows.append(
            {
                "run_name": run_name,
                MODEL_ID_COL: model_id,
                SCENARIO_COL: scenario,
                "seed": seed,
                "WeightPath": str(best_pt),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No se encontraron archivos best.pt en {R4_RUNS_DIR}")

    df = pd.DataFrame(rows).drop_duplicates(subset=["run_name"]).reset_index(drop=True)
    logger.info(f"Pesos R4 detectados: {len(df)}")
    return df


def evaluate_weights_multidomain(weights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate each best.pt from RONDA_4 on all configured test domains.
    No input CSV is used as source of truth.
    """
    cache_cols = [DATASET_COL, MODEL_ID_COL, SCENARIO_COL, "seed", "run_name", "WeightPath", *METRICS]
    eval_rows: List[Dict[str, object]] = []
    done_keys = set()

    if EVAL_USE_CACHE and RAW_EVAL_CSV.exists():
        cached = pd.read_csv(RAW_EVAL_CSV)
        missing_cols = set(cache_cols) - set(cached.columns)
        if not missing_cols:
            eval_rows.extend(cached[cache_cols].to_dict("records"))
            done_keys = {(str(r[DATASET_COL]), str(r["run_name"])) for r in eval_rows}
            logger.info(f"Cache de evaluaciÃ³n cargado: {RAW_EVAL_CSV} ({len(done_keys)} combinaciones)")

    device = 0 if torch.cuda.is_available() else "cpu"
    for dataset_name, dataset_path in TEST_DATASETS.items():
        if not dataset_path.exists():
            logger.warning(f"Dataset no encontrado, se omite: {dataset_name} -> {dataset_path}")
            continue

        yaml_path = create_test_yaml(dataset_path, TMP_YAML_DIR / f"r4_eval_{dataset_name.lower()}.yaml")
        for _, wr in weights_df.iterrows():
            key = (dataset_name, str(wr["run_name"]))
            if key in done_keys:
                continue

            model = None
            try:
                clean_gpu()
                model = YOLO(str(wr["WeightPath"]))
                metrics = model.val(
                    data=str(yaml_path),
                    split="test",
                    batch=EVAL_BATCH,
                    imgsz=EVAL_IMGSZ,
                    device=device,
                    verbose=False,
                    plots=False,
                )
                m = extract_metrics(metrics)
                row = {
                    DATASET_COL: dataset_name,
                    MODEL_ID_COL: wr[MODEL_ID_COL],
                    SCENARIO_COL: wr[SCENARIO_COL],
                    "seed": int(wr["seed"]),
                    "run_name": str(wr["run_name"]),
                    "WeightPath": str(wr["WeightPath"]),
                    "mAP50-95": m["mAP50-95"],
                    "mAP50": m["mAP50"],
                    "Precision": m["Precision"],
                    "Recall": m["Recall"],
                }
                eval_rows.append(row)
                done_keys.add(key)
                logger.info(
                    f"Eval OK {dataset_name} | {wr['run_name']} -> mAP50-95={m['mAP50-95']:.4f}"
                )
                pd.DataFrame(eval_rows, columns=cache_cols).to_csv(RAW_EVAL_CSV, index=False)
            except Exception as e:
                logger.error(f"Error evaluando {dataset_name} | {wr['run_name']}: {e}")
            finally:
                if model is not None:
                    del model
                clean_gpu()

    if not eval_rows:
        raise RuntimeError("No se obtuvo ningÃºn resultado de evaluaciÃ³n desde los pesos R4.")

    out = pd.DataFrame(eval_rows, columns=cache_cols)
    out.to_csv(RAW_EVAL_CSV, index=False)
    return out


def aggregate_over_seeds(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Implements M~(i,d) = mean_s M(i,d,s), then keeps one row per (scenario, dataset).
    """
    grouped = (
        df_raw.groupby([MODEL_ID_COL, SCENARIO_COL, DATASET_COL], as_index=False)[METRICS]
        .mean(numeric_only=True)
        .copy()
    )
    return grouped


def build_seed_coverage(df_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_id, scenario), g in df_raw.groupby([MODEL_ID_COL, SCENARIO_COL]):
        seeds = sorted({int(s) for s in g["seed"].dropna().tolist()})
        missing = sorted(EXPECTED_R4_SEEDS - set(seeds))
        rows.append(
            {
                MODEL_ID_COL: model_id,
                SCENARIO_COL: scenario,
                "seeds_present": ",".join(str(s) for s in seeds),
                "n_seeds_present": len(seeds),
                "missing_expected_seeds": ",".join(str(s) for s in missing),
            }
        )
    return pd.DataFrame(rows).sort_values([MODEL_ID_COL, SCENARIO_COL]).reset_index(drop=True)


def _metric_moments(values: pd.Series) -> Dict[str, float]:
    x = values.astype(float).dropna()
    if x.empty:
        return {
            "n_reps": 0,
            "mean": float("nan"),
            "var": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
            "skew": float("nan"),
            "kurtosis": float("nan"),
            "moment3": float("nan"),
            "moment4": float("nan"),
        }
    mu = float(x.mean())
    c = x - mu
    return {
        "n_reps": int(x.shape[0]),
        "mean": mu,
        "var": float(x.var(ddof=1)) if x.shape[0] > 1 else 0.0,
        "std": float(x.std(ddof=1)) if x.shape[0] > 1 else 0.0,
        "min": float(x.min()),
        "median": float(x.median()),
        "max": float(x.max()),
        "skew": float(x.skew()),
        "kurtosis": float(x.kurt()),
        "moment3": float((c**3).mean()),
        "moment4": float((c**4).mean()),
    }


def compute_seed_moments_wide(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, object] = {col: key for col, key in zip(group_cols, keys)}
        n_rep_values = []
        for m in metric_cols:
            moments = _metric_moments(g[m])
            n_rep_values.append(moments["n_reps"])
            for stat_name, stat_val in moments.items():
                if stat_name == "n_reps":
                    continue
                row[f"{m}_{stat_name}"] = stat_val
        row["n_reps"] = int(min(n_rep_values)) if n_rep_values else 0
        rows.append(row)
    return pd.DataFrame(rows)


def compute_group_stat_moments(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Moments by dataset: repetitions are seeds for each (model, domain).
    by_dataset = compute_seed_moments_wide(
        df_raw,
        group_cols=[MODEL_ID_COL, SCENARIO_COL, DATASET_COL],
        metric_cols=METRICS,
    ).sort_values([MODEL_ID_COL, SCENARIO_COL, DATASET_COL]).reset_index(drop=True)

    # Global moments by model: first aggregate each seed across domains.
    by_seed_global = (
        df_raw.groupby([MODEL_ID_COL, SCENARIO_COL, "seed"], as_index=False)[METRICS]
        .mean(numeric_only=True)
        .copy()
    )
    global_moments = compute_seed_moments_wide(
        by_seed_global,
        group_cols=[MODEL_ID_COL, SCENARIO_COL],
        metric_cols=METRICS,
    ).sort_values([MODEL_ID_COL, SCENARIO_COL]).reset_index(drop=True)

    return by_dataset, global_moments


def build_summary(df_round: pd.DataFrame) -> pd.DataFrame:
    s = df_round.groupby([MODEL_ID_COL, SCENARIO_COL])[METRICS].agg(["mean", "min"])
    s.columns = [f"{m}_{'Promedio' if agg == 'mean' else 'PeorCaso'}" for (m, agg) in s.columns]
    col_order = []
    for m in METRICS:
        col_order += [f"{m}_Promedio", f"{m}_PeorCaso"]
    return s[col_order].reset_index()


def display_label(model_id: object, scenario: object) -> str:
    return f"{model_id}|{scenario}"


def plot_dataset_bars(df_round: pd.DataFrame) -> None:
    for dataset in sorted(df_round[DATASET_COL].dropna().unique().tolist()):
        df_ds = df_round[df_round[DATASET_COL] == dataset].copy().sort_values("mAP50-95", ascending=True)
        scenarios = [
            display_label(mid, sc) for mid, sc in zip(df_ds[MODEL_ID_COL].tolist(), df_ds[SCENARIO_COL].tolist())
        ]
        x = np.arange(len(scenarios))
        bw = 0.18
        plt.figure(figsize=(max(10, 1.2 * len(scenarios)), 6))
        for i, m in enumerate(METRICS):
            bars = plt.bar(x + i * bw, df_ds[m].astype(float).values, bw, label=METRIC_LABELS[m])
            for bar in bars:
                h = float(bar.get_height())
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )
        plt.title(f"{R4_NAME} - {dataset}")
        plt.xlabel("Escenario")
        plt.ylabel("Score")
        plt.ylim(0, 1.05)
        plt.xticks(x + bw * (len(METRICS) - 1) / 2, scenarios, rotation=30, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGS_DIR / f"{R4_NAME}_bars_{dataset}.png", dpi=200, bbox_inches="tight")
        plt.close()


def plot_abs_summary_heatmap(df_summary: pd.DataFrame) -> None:
    cols = [c for c in df_summary.columns if c not in {MODEL_ID_COL, SCENARIO_COL}]
    mat = df_summary[cols].to_numpy(dtype=float) * PERCENT_SCALE
    labels = [
        display_label(mid, sc) for mid, sc in zip(df_summary[MODEL_ID_COL].tolist(), df_summary[SCENARIO_COL].tolist())
    ]
    plt.figure(figsize=(max(10, 0.45 * len(cols)), max(4, 0.45 * len(labels))))
    im = plt.imshow(mat, aspect="auto", cmap=ABS_CMAP)
    plt.title(f"{R4_NAME} - Heatmap Resumen (Absoluto, %)")
    plt.xticks(range(len(cols)), cols, rotation=30, ha="right", fontsize=11)
    plt.yticks(range(len(labels)), labels, fontsize=11)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f"{mat[i, j]:.3f}%", ha="center", va="center", fontsize=12, fontweight="bold")
    cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / f"{R4_NAME}_heatmap_resumen_abs.png", dpi=200, bbox_inches="tight")
    plt.close()


def delta_matrix(values: pd.Series) -> pd.DataFrame:
    v = values.astype(float).values
    d = (v[:, None] - v[None, :]) * DELTA_SCALE
    return pd.DataFrame(d, index=values.index, columns=values.index)


def plot_delta_heatmap(df_delta: pd.DataFrame, metric: str, dataset: str) -> None:
    mat = df_delta.to_numpy(dtype=float)
    labels = df_delta.index.tolist()
    plt.figure(figsize=(max(6, 0.6 * len(labels)), max(5, 0.6 * len(labels))))
    im = plt.imshow(mat, aspect="auto", cmap=DELTA_CMAP)
    plt.title(f"{R4_NAME} - Deltas {metric} - {dataset}")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=11)
    plt.yticks(range(len(labels)), labels, fontsize=11)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f"{mat[i, j]:+.3f}%", ha="center", va="center", fontsize=12, fontweight="bold")
    cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / f"{R4_NAME}_deltas_{metric}_{dataset}.png", dpi=200, bbox_inches="tight")
    plt.close()


def generate_deltas(df_round: pd.DataFrame) -> None:
    order = [s for s in R4_FINALISTS if s in set(df_round[SCENARIO_COL].unique().tolist())]
    for metric in METRICS:
        for dataset in sorted(df_round[DATASET_COL].dropna().unique().tolist()):
            df_ds = df_round[df_round[DATASET_COL] == dataset]
            s = df_ds.set_index(SCENARIO_COL)[metric].reindex(order)
            if s.isna().any():
                continue
            df_delta = delta_matrix(s)
            df_delta.to_csv(FIGS_DIR / f"{R4_NAME}_deltas_{metric}_{dataset}.csv")
            plot_delta_heatmap(df_delta, metric, dataset)


def plot_gating_threshold(scored_all: pd.DataFrame, tau_r: float) -> None:
    if scored_all.empty or "Recall_PeorCaso" not in scored_all.columns:
        return
    df = scored_all.copy().sort_values("Recall_PeorCaso", ascending=True)
    df["label"] = [display_label(mid, sc) for mid, sc in zip(df[MODEL_ID_COL], df[SCENARIO_COL])]
    colors = ["tab:green" if bool(v) else "tab:red" for v in df["gating_pass"]]

    plt.figure(figsize=(12, max(5, 0.5 * len(df))))
    bars = plt.barh(df["label"], df["Recall_PeorCaso"], color=colors)
    if not np.isnan(tau_r):
        plt.axvline(tau_r, color="black", linestyle="--", linewidth=1.8, label=f"tau_min={tau_r:.4f}")
        plt.legend(loc="lower right")

    plt.title(f"{R4_NAME} - Gating por Recall_min")
    plt.xlabel("Recall_min")
    plt.ylabel("Escenario")
    plt.grid(axis="x", linestyle="--", alpha=0.35)
    for bar in bars:
        w = float(bar.get_width())
        y = bar.get_y() + bar.get_height() / 2
        plt.text(w + 0.0015, y, f"{w:.4f}", va="center", ha="left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / f"{R4_NAME}_gating_threshold.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_recall_min_vs_tau_min(scored_all: pd.DataFrame, tau_r: float) -> None:
    if scored_all.empty or "Recall_PeorCaso" not in scored_all.columns:
        return

    df = scored_all.copy().sort_values("Recall_PeorCaso", ascending=True)
    df["label"] = [display_label(mid, sc) for mid, sc in zip(df[MODEL_ID_COL], df[SCENARIO_COL])]
    colors = ["tab:green" if bool(v) else "tab:red" for v in df["gating_pass"]]

    plt.figure(figsize=(12, max(5, 0.5 * len(df))))
    bars = plt.barh(df["label"], df["Recall_PeorCaso"], color=colors, edgecolor="white")
    if not np.isnan(tau_r):
        plt.axvline(tau_r, color="#8b0000", linestyle="--", linewidth=1.8, label=f"tau_min={tau_r:.4f}")
        plt.legend(loc="lower right")

    plt.title(f"{R4_NAME} - Recall_min vs tau_min")
    plt.xlabel("Recall_min")
    plt.ylabel("Escenario")
    plt.grid(axis="x", linestyle="--", alpha=0.35)
    for bar in bars:
        w = float(bar.get_width())
        y = bar.get_y() + bar.get_height() / 2
        plt.text(w + 0.0015, y, f"{w:.4f}", va="center", ha="left", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / f"{R4_NAME}_recall_min_vs_tau_min.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_score_ranking(scored_pass: pd.DataFrame) -> None:
    if scored_pass.empty:
        return
    df = scored_pass.copy().sort_values("Score_global", ascending=True)
    df["label"] = [display_label(mid, sc) for mid, sc in zip(df[MODEL_ID_COL], df[SCENARIO_COL])]
    plt.figure(figsize=(10, max(4, 0.5 * len(df))))
    bars = plt.barh(df["label"], df["Score_global"], color="tab:blue")
    plt.title(f"{R4_NAME} - Ranking final por Score_global")
    plt.xlabel("Score_global")
    plt.ylabel("Escenario")
    plt.grid(axis="x", linestyle="--", alpha=0.35)
    for bar in bars:
        w = float(bar.get_width())
        y = bar.get_y() + bar.get_height() / 2
        plt.text(w + 0.002, y, f"{w:.4f}", va="center", ha="left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / f"{R4_NAME}_score_ranking.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_final_selection_chart(scored_all: pd.DataFrame, tau_r: float, winner_label: str | None) -> None:
    """
    Combined final chart (R7-style):
    - Horizontal bars of Score_global
    - Recall min annotation
    - Vertical gating threshold tau_min
    - Parameter box (alpha, tau, anchor)
    """
    if scored_all.empty:
        return

    df = scored_all.copy().sort_values("Score_global", ascending=True)
    y = np.arange(len(df))
    labels = [display_label(mid, sc) for mid, sc in zip(df[MODEL_ID_COL], df[SCENARIO_COL])]
    winner_colors = [
        "#1f77b4" if winner_label is not None and s == winner_label else "#d9d9d9" for s in labels
    ]
    gating_colors = ["tab:green" if bool(v) else "tab:red" for v in df["gating_pass"]]

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(15, max(4, 0.7 * len(df))), sharey=True, gridspec_kw={"width_ratios": [1, 1.2]}
    )

    # Left panel: gating variable (Recall_min) with tau_min threshold
    bars_l = ax_l.barh(y, df["Recall_PeorCaso"], color=gating_colors, edgecolor="white")
    if not np.isnan(tau_r):
        ax_l.axvline(tau_r, color="#8b0000", linestyle="--", linewidth=1.8, label=f"tau_min={tau_r:.4f}")
        ax_l.legend(loc="lower right", fontsize=8)
    ax_l.set_title("Gating por Recall_min")
    ax_l.set_xlabel("Recall_min")
    ax_l.set_yticks(y)
    ax_l.set_yticklabels(labels)
    ax_l.grid(axis="x", linestyle="--", alpha=0.30)
    for bar in bars_l:
        w = float(bar.get_width())
        yy = bar.get_y() + bar.get_height() / 2
        ax_l.text(w + 0.0015, yy, f"{w:.4f}", va="center", ha="left", fontsize=8)

    # Right panel: decision variable (Score_global)
    bars_r = ax_r.barh(y, df["Score_global"], color=winner_colors, edgecolor="white")
    ax_r.set_title("Ranking por Score_global")
    ax_r.set_xlabel("Score Global Ponderado")
    ax_r.grid(axis="x", linestyle="--", alpha=0.30)
    for bar in bars_r:
        w = float(bar.get_width())
        yy = bar.get_y() + bar.get_height() / 2
        ax_r.text(w + 0.002, yy, f"{w:.4f}", va="center", ha="left", fontsize=9, weight="bold")

    # Parameter box on right panel
    param_lines = [
        "ParÃ¡metros DoE:",
        f"Alpha: {ALPHA:.2f}",
        f"tau_min: {tau_r:.4f}" if not np.isnan(tau_r) else "tau_min: N/A",
        f"Gating Anchor: {GATING_ANCHOR}" if GATING_MODE == "anchor" else "Gating: none",
    ]
    ax_r.text(
        0.99,
        0.03,
        "\n".join(param_lines),
        transform=ax_r.transAxes,
        fontsize=8,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bdbdbd", alpha=0.95),
    )

    fig.suptitle(f"{R4_NAME}: EvaluaciÃ³n Final - SelecciÃ³n del Candidato Ã“ptimo", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGS_DIR / f"{R4_NAME}_final_selection_chart.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def apply_gating(df_summary: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    if GATING_MODE == "none":
        out = df_summary.copy()
        out["gating_pass"] = True
        return out, float("nan")
    if GATING_MODE != "anchor":
        raise ValueError("GATING_MODE debe ser 'anchor' o 'none'.")
    if GATING_ANCHOR not in df_summary[SCENARIO_COL].values:
        raise ValueError(f"Ancla '{GATING_ANCHOR}' no estÃ¡ en R4_FINALISTS.")
    r_anchor = float(df_summary.loc[df_summary[SCENARIO_COL] == GATING_ANCHOR, "Recall_PeorCaso"].iloc[0])
    tau = GATING_FACTOR * r_anchor
    out = df_summary.copy()
    out["gating_pass"] = out["Recall_PeorCaso"] >= tau
    return out, tau


def compute_scores(df_summary: pd.DataFrame) -> pd.DataFrame:
    if abs(sum(WEIGHTS.values()) - 1.0) > 1e-6:
        raise ValueError("WEIGHTS debe sumar 1.0")
    out = df_summary.copy()
    for m in METRICS:
        out[f"S_{m}"] = ALPHA * out[f"{m}_Promedio"].astype(float) + (1.0 - ALPHA) * out[
            f"{m}_PeorCaso"
        ].astype(float)
    out["Score_global"] = 0.0
    for m, w in WEIGHTS.items():
        out["Score_global"] += float(w) * out[f"S_{m}"]
    return out


def make_score_table(scored_all: pd.DataFrame, tau_r: float | None) -> pd.DataFrame:
    rows = []
    for _, r in scored_all.iterrows():
        row = {
            "ModelID": r.get(MODEL_ID_COL, "NA"),
            "Escenario": r[SCENARIO_COL],
            "ALPHA": ALPHA,
            "Formula_S_m": "S_m = Î±Â·Promedio + (1-Î±)Â·Min",
            "Formula_Sglobal": "Score_global = Î£ w_mÂ·S_m",
            "tau_R": np.nan if tau_r is None or (isinstance(tau_r, float) and np.isnan(tau_r)) else float(tau_r),
            "Recall_PeorCaso": float(r["Recall_PeorCaso"]),
            "gating_pass": bool(r.get("gating_pass", True)),
        }

        contrib_sum = 0.0
        for m in METRICS:
            prom = float(r[f"{m}_Promedio"])
            peor = float(r[f"{m}_PeorCaso"])
            s_m = float(r[f"S_{m}"])
            w = float(WEIGHTS[m])
            contrib = w * s_m
            contrib_sum += contrib

            row[f"{m}_Promedio"] = prom
            row[f"{m}_PeorCaso"] = peor
            row[f"w_{m}"] = w
            row[f"S_{m}"] = s_m
            row[f"wÂ·S_{m}"] = contrib

        row["Score_global_recalc"] = contrib_sum
        row["Score_global_script"] = float(r["Score_global"])
        rows.append(row)

    df_out = pd.DataFrame(rows)
    front = ["ModelID", "Escenario", "ALPHA", "Formula_S_m", "Formula_Sglobal", "tau_R", "Recall_PeorCaso", "gating_pass"]
    metric_cols = []
    for m in METRICS:
        metric_cols += [f"{m}_Promedio", f"{m}_PeorCaso", f"w_{m}", f"S_{m}", f"wÂ·S_{m}"]
    tail = ["Score_global_recalc", "Score_global_script"]
    return df_out[front + metric_cols + tail]


def run_r4_final() -> None:
    if not R4_FINALISTS:
        raise ValueError("R4_FINALISTS estÃ¡ vacÃ­o. Define finalistas de R0-R3.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_YAML_DIR.mkdir(parents=True, exist_ok=True)

    weights_df = discover_r4_weights()
    weights_df = weights_df[weights_df[SCENARIO_COL].isin(R4_FINALISTS)].copy()
    if weights_df.empty:
        raise ValueError("No hay pesos en RONDA_4 para los escenarios definidos en R4_FINALISTS.")

    # Canonical model id per scenario for stable exports (MCDM/gating tables).
    scenario_to_model = (
        weights_df[[MODEL_ID_COL, SCENARIO_COL]]
        .drop_duplicates()
        .sort_values([SCENARIO_COL, MODEL_ID_COL])
        .groupby(SCENARIO_COL, as_index=False)
        .first()
        .set_index(SCENARIO_COL)[MODEL_ID_COL]
        .to_dict()
    )

    df_raw = evaluate_weights_multidomain(weights_df)
    df_r4_raw = df_raw[df_raw[SCENARIO_COL].isin(R4_FINALISTS)].copy()

    present_scenarios = set(df_r4_raw[SCENARIO_COL].unique().tolist())
    missing = [s for s in R4_FINALISTS if s not in present_scenarios]
    if missing:
        logger.warning(f"Finalistas definidos sin pesos/evaluaciones en R4: {missing}")
    if not present_scenarios:
        raise ValueError("No se encontraron escenarios evaluados para R4.")

    coverage = build_seed_coverage(df_r4_raw)
    coverage.to_csv(OUT_DIR / f"{R4_NAME}_seed_coverage.csv", index=False)
    for _, r in coverage.iterrows():
        if r["missing_expected_seeds"]:
            logger.warning(
                f"{r.get(MODEL_ID_COL, 'NA')}|{r[SCENARIO_COL]}: semillas faltantes para R4 esperado {sorted(EXPECTED_R4_SEEDS)} -> {r['missing_expected_seeds']}"
            )

    moments_by_dataset, moments_global = compute_group_stat_moments(df_r4_raw)
    moments_by_dataset.to_csv(OUT_DIR / f"{R4_NAME}_seed_moments_by_dataset.csv", index=False)
    moments_global.to_csv(OUT_DIR / f"{R4_NAME}_seed_moments_global.csv", index=False)

    # Core update for Section E:
    #   M~(i,d) = mean_s M(i,d,s)
    # This leaves one row per (scenario, domain).
    df_r4 = aggregate_over_seeds(df_r4_raw)

    order_map = {s: i for i, s in enumerate(R4_FINALISTS)}
    df_r4["order"] = df_r4[SCENARIO_COL].map(order_map)
    df_r4 = df_r4.sort_values(["order", DATASET_COL]).drop(columns=["order"])

    df_r4_raw.to_csv(OUT_DIR / f"{R4_NAME}_raw_multiseed.csv", index=False)
    df_r4.to_csv(OUT_DIR / f"{R4_NAME}.csv", index=False)
    summary = build_summary(df_r4)
    summary[MODEL_ID_COL] = summary[SCENARIO_COL].map(scenario_to_model).fillna(summary[MODEL_ID_COL])
    summary.to_csv(OUT_DIR / f"{R4_NAME}_resumen.csv", index=False)

    plot_dataset_bars(df_r4)
    plot_abs_summary_heatmap(summary)
    generate_deltas(df_r4)

    gated, tau = apply_gating(summary)
    scored_all = compute_scores(gated)
    scored_all[MODEL_ID_COL] = scored_all[SCENARIO_COL].map(scenario_to_model).fillna(scored_all[MODEL_ID_COL])
    scored_all.to_csv(OUT_DIR / f"{R4_NAME}_scored_all.csv", index=False)
    score_table = make_score_table(scored_all, tau_r=tau)
    score_table.to_csv(OUT_DIR / f"{R4_NAME}_score_table.csv", index=False)

    scored_pass = scored_all[scored_all["gating_pass"]].copy()
    scored_pass = scored_pass.sort_values(
        by=["Score_global", "Recall_PeorCaso", "mAP50-95_PeorCaso", "S_mAP50-95"],
        ascending=[False, False, False, False],
    )
    scored_pass.to_csv(OUT_DIR / f"{R4_NAME}_ranking.csv", index=False)
    plot_gating_threshold(scored_all, tau)
    plot_recall_min_vs_tau_min(scored_all, tau)
    plot_score_ranking(scored_pass)

    winner = scored_pass.iloc[0][SCENARIO_COL] if not scored_pass.empty else None
    winner_model_id = scored_pass.iloc[0][MODEL_ID_COL] if not scored_pass.empty else None
    winner_label = (
        display_label(winner_model_id, winner) if winner is not None and winner_model_id is not None else None
    )
    plot_final_selection_chart(scored_all, tau, winner_label)
    with open(OUT_DIR / f"{R4_NAME}_resultado.txt", "w", encoding="utf-8") as f:
        f.write(f"{R4_NAME} - Seleccion final\n")
        f.write("=============================\n\n")
        f.write(f"Fuente de datos: evaluacion directa de pesos en {R4_RUNS_DIR}\n")
        f.write(f"Finalistas: {R4_FINALISTS}\n")
        f.write(f"ModelIDs finalistas: {sorted(set(weights_df[MODEL_ID_COL].astype(str).tolist()))}\n")
        f.write(f"Semillas esperadas R4: {sorted(EXPECTED_R4_SEEDS)}\n")
        f.write(f"ALPHA = {ALPHA}\n")
        f.write(f"WEIGHTS = {WEIGHTS}\n")
        f.write(f"GATING_MODE = {GATING_MODE}\n")
        if GATING_MODE == "anchor":
            f.write(f"GATING_ANCHOR = {GATING_ANCHOR}\n")
            f.write(f"GATING_FACTOR = {GATING_FACTOR}\n")
            f.write(f"tau_min = {tau:.6f}\n")
        f.write(f"\nGanador: {winner_label}\n")

    logger.info(f"R4 final completado en: {OUT_DIR}")
    print(f"R4 final completado en: {OUT_DIR}")
    print(f"Ganador: {winner_label}")


if __name__ == "__main__":
    run_r4_final()
