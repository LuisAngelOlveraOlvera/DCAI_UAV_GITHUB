"""
23 DISTANCE DEGRADATION ANALYSIS (R4 finalists)
-----------------------------------------------------------------------
Analyze finalist robustness by:
1) reading distance-binned evaluation outputs
2) measuring performance drop across distance ranges
3) ranking finalists under degradation criteria
-----------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import utils


logger = utils.setup_logger(
    "Distance_R4",
    log_file=str(config.LOGS_DIR / "distance_degradation_r4.log"),
)

# ===============================
# CONFIG
# ===============================
CSV_PATH = config.EXPORTS_DIR / "resultados_por_distancia.csv"
OUT_DIR = config.EXPORTS_DIR / "distance_r4_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COL = "Modelo"
DIST_COL = "Distancia"
METRICS = ["mAP50-95", "mAP50", "Precision", "Recall"]

# Canonical finalists aligned with 19_doe_r4_final.py
R4_FINALISTS = [
    "B_Raw_0",
    "P_Raw_0",
    "Z_COCO",
    "H_pH_JS_0",
    "H_pH_JS_40",
    "H_pH_JR_15",
    "H_pH_JR_40",
    "H_JR_0",
    "P_pH_JR_40",
]

# N-index to canonical mapping used in this repository
N_TO_CANONICAL = {
    "N1": "Z_COCO",
    "N2": "B_Raw_0",
    "N3": "H_Raw_0",
    "N4": "H_pH_0",
    "N5": "H_JR_0",
    "N6": "H_pH_JR_0",
    "N7": "H_pH_JS_0",
    "N8": "H_pH_JR_15",
    "N9": "H_pH_JR_40",
    "N10": "H_pH_JS_40",
    "N11": "P_Raw_0",
    "N12": "P_pH_JR_0",
    "N13": "P_pH_JR_15",
    "N14": "P_pH_JR_40",
}

# Distance robustness score
DIST_SCORE_WEIGHTS: Dict[str, float] = {"mAP50-95": 0.60, "Recall": 0.40}

# Heatmaps
ABS_HEATMAP_METRICS = ["mAP50-95", "Recall"]
ABS_CMAP = "Greens"
DELTA_CMAP = "RdYlGn"
HEATMAP_DPI = 200

# Deltas in percentage points
DELTA_SCALE = 100.0
DELTA_FMT = "{:+.2f}%"
BASELINE_NAME_FOR_DELTAS = "B_Raw_0"


def normalize_model_name(raw_name: str) -> str:
    name = str(raw_name).strip()
    if name in R4_FINALISTS:
        return name
    for n_id, canonical in sorted(
        N_TO_CANONICAL.items(), key=lambda item: len(item[0]), reverse=True
    ):
        # Match repository IDs such as "N1_" without letting "N1" swallow "N10"/"N11".
        if name == n_id or name.startswith(f"{n_id}_"):
            return canonical
    return name


def trapezoid_auc(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapz(y, x))


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    a, _b = np.polyfit(x, y, 1)
    return float(a)


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b != 0 else np.nan


def coerce_metric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.replace("%", "", regex=False)
    values = pd.to_numeric(cleaned, errors="coerce")
    pct_mask = series.astype(str).str.contains("%", regex=False, na=False)
    values.loc[pct_mask] = values.loc[pct_mask] / 100.0
    return values


def set_dynamic_ylim(
    values: List[float], lower_bound: float = 0.0, upper_bound: float = 1.05
) -> None:
    clean_values = [float(v) for v in values if pd.notna(v)]
    if not clean_values:
        return

    y_min = min(clean_values)
    y_max = max(clean_values)
    margin = max(0.01, (y_max - y_min) * 0.12)
    lower = max(lower_bound, y_min - margin)
    upper = min(upper_bound, y_max + margin)

    if lower == upper:
        upper = min(upper_bound, lower + 0.05)

    plt.ylim(lower, upper)


def build_canonical_fullname_map(df_in: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for canonical in sorted(df_in["ModeloCanonical"].dropna().unique().tolist()):
        raw_names = sorted(df_in.loc[df_in["ModeloCanonical"] == canonical, MODEL_COL].unique().tolist())
        if raw_names:
            mapping[canonical] = raw_names[0]
    return mapping


def build_score_map(ranking_df: pd.DataFrame) -> Dict[str, float]:
    return dict(zip(ranking_df["Modelo"], ranking_df["Score_dist"]))


def format_display_label(model_name: str, score: float, multiline: bool = False) -> str:
    sep = "\n" if multiline else " | "
    return f"{model_name}{sep}Score={score:.3f}"


def format_metric_legend(model_name: str, df_model: pd.DataFrame, metric: str) -> str:
    parts = [
        f"{int(dist)}m={value:.4f}"
        for dist, value in zip(df_model[DIST_COL].tolist(), df_model[metric].tolist())
    ]
    return f"{model_name}\n" + " | ".join(parts)


def compute_distance_derivatives(
    df_in: pd.DataFrame, model_col: str, model_order: List[str]
) -> pd.DataFrame:
    rows = []
    for m in model_order:
        df_m = df_in[df_in[model_col] == m].sort_values(DIST_COL)
        if df_m.empty:
            continue
        x = df_m[DIST_COL].values.astype(float)
        span = float(x.max() - x.min()) if x.max() != x.min() else 1.0

        for metric in METRICS:
            y = df_m[metric].values.astype(float)
            auc = trapezoid_auc(x, y)
            auc_norm = auc / span
            slope = linear_slope(x, y)
            y_min = float(df_m[df_m[DIST_COL] == x.min()][metric].iloc[0])
            y_max = float(df_m[df_m[DIST_COL] == x.max()][metric].iloc[0])
            rows.append(
                {
                    "Modelo": m,
                    "Metrica": metric,
                    "AUC": auc,
                    "AUC_norm": auc_norm,
                    "Slope_per_m": slope,
                    "Value_minDist": y_min,
                    "Value_maxDist": y_max,
                    "Drop_max_vs_min": y_max - y_min,
                    "Ratio_max_vs_min": safe_ratio(y_max, y_min),
                }
            )

    return pd.DataFrame(rows)


def build_distance_ranking(
    df_derived: pd.DataFrame, df_source: pd.DataFrame, model_col: str
) -> pd.DataFrame:
    pivot_auc = (
        df_derived[df_derived["Metrica"].isin(DIST_SCORE_WEIGHTS.keys())]
        .pivot_table(index="Modelo", columns="Metrica", values="AUC_norm", aggfunc="mean")
    )
    pivot_auc["Score_dist"] = 0.0
    for metric, w in DIST_SCORE_WEIGHTS.items():
        if metric not in pivot_auc.columns:
            raise ValueError(f"No existe AUC_norm para {metric} en pivot.")
        pivot_auc["Score_dist"] += w * pivot_auc[metric]

    max_dist = df_source[DIST_COL].max()

    def value_at_maxdist(model: str, metric: str) -> float:
        d = df_source[(df_source[model_col] == model) & (df_source[DIST_COL] == max_dist)]
        return float(d[metric].iloc[0]) if not d.empty else np.nan

    tiebreak_rows = []
    for m in pivot_auc.index:
        tiebreak_rows.append(
            {
                "Modelo": m,
                "Recall_at_maxDist": value_at_maxdist(m, "Recall"),
                "mAP50-95_at_maxDist": value_at_maxdist(m, "mAP50-95"),
            }
        )
    df_tb = pd.DataFrame(tiebreak_rows).set_index("Modelo")

    ranking = pivot_auc.join(df_tb)
    ranking = ranking.sort_values(
        by=["Score_dist", "Recall_at_maxDist", "mAP50-95_at_maxDist"],
        ascending=[False, False, False],
    ).reset_index()
    return ranking


def plot_heatmap_matrix(
    mat: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    out_png: Path,
    cmap: str,
    fmt_func,
    annot_fs: int = 10,
) -> None:
    fig_w = max(7, 0.65 * len(col_labels))
    fig_h = max(4, 0.55 * len(row_labels))

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat, aspect="auto", cmap=cmap)
    plt.title(title, fontsize=12)
    plt.xticks(range(len(col_labels)), col_labels)
    plt.yticks(range(len(row_labels)), row_labels)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = "" if np.isnan(v) else fmt_func(v)
            plt.text(j, i, txt, ha="center", va="center", fontsize=annot_fs)

    plt.colorbar(im, fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=HEATMAP_DPI, bbox_inches="tight")
    plt.close()


def run_distance_analysis() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No existe CSV de distancia: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    required = [MODEL_COL, DIST_COL] + METRICS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en CSV de distancia: {missing}")

    df[DIST_COL] = pd.to_numeric(df[DIST_COL], errors="raise")
    for metric in METRICS:
        df[metric] = coerce_metric_series(df[metric])
    df["ModeloCanonical"] = df[MODEL_COL].map(normalize_model_name)
    all_models = sorted(df[MODEL_COL].unique().tolist())

    df_sel = df[df["ModeloCanonical"].isin(R4_FINALISTS)].copy()
    if df_sel.empty:
        raise ValueError(
            "El filtro de finalistas dejÃ³ el dataframe vacÃ­o. Revisa nombres en resultados_por_distancia.csv."
        )

    expected_dist = sorted(df_sel[DIST_COL].unique().tolist())
    problems = []
    for m in R4_FINALISTS:
        dists_m = sorted(df_sel[df_sel["ModeloCanonical"] == m][DIST_COL].unique().tolist())
        if dists_m != expected_dist:
            problems.append((m, dists_m))
    if problems:
        logger.warning("Modelos con distancias incompletas/diferentes detectados.")
        for model, dists in problems:
            logger.warning(f"{model}: {dists}")

    canonical_fullname_map = build_canonical_fullname_map(df_sel)
    df_derived = compute_distance_derivatives(df_sel, "ModeloCanonical", R4_FINALISTS)
    df_derived.to_csv(OUT_DIR / "distance_metrics_derived_long.csv", index=False)
    ranking = build_distance_ranking(df_derived, df_sel, "ModeloCanonical")
    ranking.to_csv(OUT_DIR / "ranking_distance_final.csv", index=False)
    r4_score_map = build_score_map(ranking)
    r4_display_map = {
        model: format_display_label(canonical_fullname_map.get(model, model), r4_score_map[model])
        for model in ranking["Modelo"].tolist()
    }
    r4_display_multiline = {
        model: format_display_label(
            canonical_fullname_map.get(model, model), r4_score_map[model], multiline=True
        )
        for model in ranking["Modelo"].tolist()
    }

    # 1) Curvas por mÃ©trica
    for metric in METRICS:
        fig, (ax_plot, ax_legend) = plt.subplots(
            1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [3.6, 2.4]}
        )
        metric_values = []
        handles = []
        labels = []
        for m in R4_FINALISTS:
            df_m = df_sel[df_sel["ModeloCanonical"] == m].sort_values(DIST_COL)
            if df_m.empty:
                continue
            metric_values.extend(df_m[metric].tolist())
            (line,) = ax_plot.plot(
                df_m[DIST_COL].values,
                df_m[metric].values,
                marker="o",
                label=format_metric_legend(canonical_fullname_map.get(m, m), df_m, metric),
            )
            handles.append(line)
            labels.append(format_metric_legend(canonical_fullname_map.get(m, m), df_m, metric))
        ax_plot.set_title(f"DegradaciÃ³n por distancia - {metric}")
        ax_plot.set_xlabel("Distancia (m)")
        ax_plot.set_ylabel(metric)
        plt.sca(ax_plot)
        set_dynamic_ylim(metric_values)
        ax_plot.grid(True, linestyle="--", alpha=0.40)
        ax_legend.axis("off")
        ax_legend.legend(handles, labels, loc="center left", fontsize=8, frameon=True)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"curvas_{metric}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 3b) Global ranking and top 5 using all models from the CSV with the same logic
    df_derived_all = compute_distance_derivatives(df, MODEL_COL, all_models)
    df_derived_all.to_csv(OUT_DIR / "distance_metrics_all_models_long.csv", index=False)

    ranking_all = build_distance_ranking(df_derived_all, df, MODEL_COL)
    ranking_all.to_csv(OUT_DIR / "ranking_distance_all_models.csv", index=False)
    all_score_map = build_score_map(ranking_all)

    top5_models = ranking_all.head(5)["Modelo"].tolist()
    df_top5 = df[df[MODEL_COL].isin(top5_models)].copy()

    for metric in METRICS:
        fig, (ax_plot, ax_legend) = plt.subplots(
            1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [3.6, 2.4]}
        )
        top5_values = []
        handles = []
        labels = []
        for m in top5_models:
            df_m = df_top5[df_top5[MODEL_COL] == m].sort_values(DIST_COL)
            if df_m.empty:
                continue
            top5_values.extend(df_m[metric].tolist())
            (line,) = ax_plot.plot(
                df_m[DIST_COL].values,
                df_m[metric].values,
                marker="o",
                linewidth=2,
                label=format_metric_legend(m, df_m, metric),
            )
            handles.append(line)
            labels.append(format_metric_legend(m, df_m, metric))
        ax_plot.set_title(f"Top 5 global por robustez - {metric}")
        ax_plot.set_xlabel("Distancia (m)")
        ax_plot.set_ylabel(metric)
        plt.sca(ax_plot)
        set_dynamic_ylim(top5_values)
        ax_plot.grid(True, linestyle="--", alpha=0.40)
        ax_legend.axis("off")
        ax_legend.legend(handles, labels, loc="center left", fontsize=9, frameon=True)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"top5_all_models_{metric}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 4) Heatmaps absolutos
    for metric in ABS_HEATMAP_METRICS:
        pv = (
            df_sel.pivot_table(index="ModeloCanonical", columns=DIST_COL, values=metric, aggfunc="mean")
            .reindex(R4_FINALISTS)
            .reindex(expected_dist, axis=1)
        )
        pv.to_csv(OUT_DIR / f"heatmap_abs_{metric}_table.csv")
        plot_heatmap_matrix(
            mat=pv.to_numpy(dtype=float),
            row_labels=[r4_display_multiline.get(model, model) for model in pv.index.tolist()],
            col_labels=[str(int(c)) for c in pv.columns.tolist()],
            title=f"Heatmap absoluto - {metric} (Modelo x Distancia)",
            out_png=OUT_DIR / f"heatmap_abs_{metric}.png",
            cmap=ABS_CMAP,
            fmt_func=lambda v: f"{v:.4f}",
            annot_fs=10,
        )

    # 5) Heatmaps delta vs baseline
    if BASELINE_NAME_FOR_DELTAS in df_sel["ModeloCanonical"].unique():
        for metric in ABS_HEATMAP_METRICS:
            pv = (
                df_sel.pivot_table(index="ModeloCanonical", columns=DIST_COL, values=metric, aggfunc="mean")
                .reindex(R4_FINALISTS)
                .reindex(expected_dist, axis=1)
            )
            baseline_row = pv.loc[BASELINE_NAME_FOR_DELTAS].copy()
            delta = (pv.sub(baseline_row, axis=1)) * DELTA_SCALE
            delta.to_csv(
                OUT_DIR / f"heatmap_delta_vs_{BASELINE_NAME_FOR_DELTAS}_{metric}_table.csv"
            )
            plot_heatmap_matrix(
                mat=delta.to_numpy(dtype=float),
                row_labels=[r4_display_multiline.get(model, model) for model in delta.index.tolist()],
                col_labels=[str(int(c)) for c in delta.columns.tolist()],
                title=f"Delta vs {BASELINE_NAME_FOR_DELTAS} - {metric} (pp) (Modelo x Distancia)",
                out_png=OUT_DIR / f"heatmap_delta_vs_{BASELINE_NAME_FOR_DELTAS}_{metric}.png",
                cmap=DELTA_CMAP,
                fmt_func=lambda v: DELTA_FMT.format(v),
                annot_fs=9,
            )
    else:
        logger.warning(
            f"No se generaron deltas: baseline '{BASELINE_NAME_FOR_DELTAS}' no encontrado en selecciÃ³n."
        )

    # 6) AUC / slope bars with annotation
    for metric in ["mAP50-95", "Recall"]:
        d = df_derived[df_derived["Metrica"] == metric].copy().sort_values("AUC_norm", ascending=True)
        d["DisplayLabel"] = d["Modelo"].map(lambda model: r4_display_multiline.get(model, model))

        plt.figure(figsize=(10, 5))
        bars = plt.bar(d["DisplayLabel"], d["AUC_norm"])
        plt.title(f"AUC_norm por Modelo - {metric}")
        plt.xlabel("Modelo")
        plt.ylabel("AUC_norm (mÃ¡s alto = mÃ¡s robusto)")
        set_dynamic_ylim(d["AUC_norm"].tolist())
        for bar in bars:
            v = float(bar.get_height())
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90,
            )
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"AUCnorm_bar_{metric}.png", dpi=200, bbox_inches="tight")
        plt.close()

        d = d.sort_values("Slope_per_m", ascending=True)
        d["DisplayLabel"] = d["Modelo"].map(lambda model: r4_display_multiline.get(model, model))
        plt.figure(figsize=(10, 5))
        bars = plt.bar(d["DisplayLabel"], d["Slope_per_m"])
        plt.title(f"Slope por Modelo - {metric}")
        plt.xlabel("Modelo")
        plt.ylabel("Slope por metro (mÃ¡s cercano a 0 = menos degradaciÃ³n)")
        slope_values = d["Slope_per_m"].tolist()
        if slope_values:
            slope_min = min(slope_values)
            slope_max = max(slope_values)
            slope_margin = max(0.002, (slope_max - slope_min) * 0.15)
            plt.ylim(slope_min - slope_margin, slope_max + slope_margin)
        for bar in bars:
            v = float(bar.get_height())
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                v,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90,
            )
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"Slope_bar_{metric}.png", dpi=200, bbox_inches="tight")
        plt.close()

    print("\nListo.")
    print(f"Salidas en: {OUT_DIR.resolve()}")
    print("\nTop ranking por distancia:")
    print(ranking.head(10).to_string(index=False))


if __name__ == "__main__":
    run_distance_analysis()
