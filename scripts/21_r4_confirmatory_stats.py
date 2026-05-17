"""
21 R4 CONFIRMATORY STATS (post-DoE analysis)
-----------------------------------------------------------------------
Analyze R4 robustness and repeatability by:
1) reading raw multi-seed confirmatory results
2) aggregating scenario, seed, and dataset summaries
3) exporting long-format statistical tables
-----------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from math import sqrt
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config
import utils


logger = utils.setup_logger(
    "R4_Confirmatory_Stats",
    log_file=str(config.LOGS_DIR / "r4_confirmatory_stats.log"),
)

R4_NAME = "R4_FINAL"
METRICS = ["mAP50-95", "mAP50", "Precision", "Recall"]
FACTOR_SCENARIO = "scenario_canonical"
FACTOR_DATASET = "Dataset"
FACTOR_SEED = "seed"
MODEL_ID_COL = "ModelID"
TWO_PCT_POINTS = 0.02
FIGS_DIRNAME = f"{R4_NAME}_confirmatory_figs"
BAR_COLOR = "#ffb347"
EDGE_COLOR = "#d7301f"
HEATMAP_CMAP = "YlOrRd"
EFFECT_TERMS = ["Scenario", "Dataset", "Seed", "Scenario:Dataset"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute confirmatory summary tables and GLM/ANOVA artifacts for R4."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=config.EXPORTS_DIR / R4_NAME / f"{R4_NAME}_raw_multiseed.csv",
        help="Input CSV with one row per Dataset x Scenario x Seed observation.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=config.EXPORTS_DIR / R4_NAME,
        help="Directory where confirmatory outputs will be written.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha used for confidence intervals (default 0.05 => 95%% CI).",
    )
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: faltan columnas requeridas: {missing}")


def _metric_slug(metric: str) -> str:
    return (
        metric.lower()
        .replace("-", "_")
        .replace("@", "")
        .replace(".", "_")
        .replace(":", "_")
    )


def _to_numeric_frame(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_r4_raw(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_csv}")

    df = pd.read_csv(input_csv)
    _require_columns(
        df,
        [FACTOR_DATASET, FACTOR_SCENARIO, FACTOR_SEED, MODEL_ID_COL, *METRICS],
        "raw_multiseed",
    )
    df = _to_numeric_frame(df, [FACTOR_SEED, *METRICS])
    df[FACTOR_DATASET] = df[FACTOR_DATASET].astype(str)
    df[FACTOR_SCENARIO] = df[FACTOR_SCENARIO].astype(str)
    df[MODEL_ID_COL] = df[MODEL_ID_COL].astype(str)
    return df


def _describe_series(values: pd.Series, alpha: float) -> Dict[str, float]:
    s = pd.to_numeric(values, errors="coerce").dropna()
    n = int(s.shape[0])
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "cv_pct": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "ci95_half": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else 0.0
    cv_pct = float((std / abs(mean)) * 100.0) if abs(mean) > 1e-12 else np.nan
    if n > 1:
        se = std / sqrt(n)
        tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
        ci_half = tcrit * se
    else:
        ci_half = 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "cv_pct": cv_pct,
        "ci95_low": mean - ci_half,
        "ci95_high": mean + ci_half,
        "ci95_half": ci_half,
        "min": float(s.min()),
        "max": float(s.max()),
    }


def summarize_groups(
    df: pd.DataFrame,
    group_cols: List[str],
    alpha: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, grp in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, object] = dict(zip(group_cols, keys))
        row["n_obs"] = int(len(grp))
        if FACTOR_DATASET in grp.columns:
            row["n_datasets"] = int(grp[FACTOR_DATASET].nunique())
        if FACTOR_SEED in grp.columns:
            row["n_seeds"] = int(grp[FACTOR_SEED].nunique())
        if MODEL_ID_COL not in group_cols:
            mids = grp[MODEL_ID_COL].dropna().astype(str).unique().tolist()
            if len(mids) == 1:
                row[MODEL_ID_COL] = mids[0]

        for metric in METRICS:
            desc = _describe_series(grp[metric], alpha=alpha)
            for stat_name, value in desc.items():
                row[f"{metric}_{stat_name}"] = value
        rows.append(row)

    return pd.DataFrame(rows)


def build_seedmean_frame(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby([FACTOR_SCENARIO, MODEL_ID_COL, FACTOR_SEED], dropna=False)[METRICS]
        .mean()
        .reset_index()
    )
    return agg


def _scenario_label_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if MODEL_ID_COL in out.columns:
        out["label"] = out[MODEL_ID_COL].astype(str) + "|" + out[FACTOR_SCENARIO].astype(str)
    else:
        out["label"] = out[FACTOR_SCENARIO].astype(str)
    return out


def build_long_for_minitab(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [FACTOR_DATASET, MODEL_ID_COL, FACTOR_SCENARIO, FACTOR_SEED]
    long_df = df[keep_cols + METRICS].melt(
        id_vars=keep_cols,
        value_vars=METRICS,
        var_name="Metric",
        value_name="Value",
    )
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
    return long_df.sort_values(
        [FACTOR_SCENARIO, FACTOR_DATASET, FACTOR_SEED, "Metric"]
    ).reset_index(drop=True)


def seed_noise_profile(summary_by_scenario_dataset: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for metric in METRICS:
        std_col = f"{metric}_std"
        cv_col = f"{metric}_cv_pct"
        ci_col = f"{metric}_ci95_half"
        vals_std = pd.to_numeric(summary_by_scenario_dataset[std_col], errors="coerce").dropna()
        vals_cv = pd.to_numeric(summary_by_scenario_dataset[cv_col], errors="coerce").dropna()
        vals_ci = pd.to_numeric(summary_by_scenario_dataset[ci_col], errors="coerce").dropna()

        mean_std = float(vals_std.mean()) if not vals_std.empty else np.nan
        mean_ci = float(vals_ci.mean()) if not vals_ci.empty else np.nan
        max_std = float(vals_std.max()) if not vals_std.empty else np.nan

        rows.append(
            {
                "Metric": metric,
                "n_cells": int(len(vals_std)),
                "mean_within_cell_std": mean_std,
                "median_within_cell_std": float(vals_std.median()) if not vals_std.empty else np.nan,
                "max_within_cell_std": max_std,
                "mean_within_cell_cv_pct": float(vals_cv.mean()) if not vals_cv.empty else np.nan,
                "mean_ci95_half_width": mean_ci,
                "delta_reference": TWO_PCT_POINTS,
                "delta_reference_pct_points": TWO_PCT_POINTS * 100.0,
                "delta_over_mean_std": float(TWO_PCT_POINTS / mean_std) if mean_std and not np.isnan(mean_std) else np.nan,
                "delta_over_mean_ci95_half": float(TWO_PCT_POINTS / mean_ci) if mean_ci and not np.isnan(mean_ci) else np.nan,
                "delta_gt_max_std": bool(TWO_PCT_POINTS > max_std) if not np.isnan(max_std) else False,
            }
        )
    return pd.DataFrame(rows)


def _ordered_categories(series: pd.Series) -> List[str]:
    return pd.Index(series.astype(str)).drop_duplicates().tolist()


def _make_dummies(series: pd.Series, prefix: str) -> pd.DataFrame:
    cat = pd.Categorical(series.astype(str), categories=_ordered_categories(series))
    return pd.get_dummies(cat, prefix=prefix, drop_first=True, dtype=float)


def _ols_fit(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    rss = float(np.sum(resid ** 2))
    y_mean = float(np.mean(y))
    tss = float(np.sum((y - y_mean) ** 2))
    df_resid = int(len(y) - rank)
    r2 = float(1.0 - rss / tss) if tss > 0 else np.nan
    return {
        "rss": rss,
        "rank": int(rank),
        "df_resid": df_resid,
        "r2": r2,
        "n_obs": int(len(y)),
    }


def _concat_blocks(blocks: OrderedDict[str, pd.DataFrame], names: List[str]) -> np.ndarray:
    mats = [blocks[name].to_numpy(dtype=float) for name in names if name in blocks]
    return np.concatenate(mats, axis=1)


def build_design_blocks(df: pd.DataFrame) -> OrderedDict[str, pd.DataFrame]:
    scenario = _make_dummies(df[FACTOR_SCENARIO], "scenario")
    dataset = _make_dummies(df[FACTOR_DATASET], "dataset")
    seed = _make_dummies(df[FACTOR_SEED].astype(str), "seed")

    interaction_cols: Dict[str, pd.Series] = {}
    for s_col in scenario.columns:
        for d_col in dataset.columns:
            interaction_cols[f"{s_col}__x__{d_col}"] = scenario[s_col] * dataset[d_col]
    interaction = pd.DataFrame(interaction_cols, index=df.index)

    blocks: OrderedDict[str, pd.DataFrame] = OrderedDict()
    blocks["Intercept"] = pd.DataFrame({"Intercept": np.ones(len(df), dtype=float)}, index=df.index)
    blocks["Scenario"] = scenario
    blocks["Dataset"] = dataset
    blocks["Seed"] = seed
    blocks["Scenario:Dataset"] = interaction
    return blocks


def sequential_anova(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, Dict[str, float]]:
    model_df = df[[FACTOR_SCENARIO, FACTOR_DATASET, FACTOR_SEED, metric]].copy().dropna()
    if model_df.empty:
        raise ValueError(f"No hay datos validos para la metrica {metric}.")

    blocks = build_design_blocks(model_df)
    y = model_df[metric].to_numpy(dtype=float)
    order = ["Intercept", "Scenario", "Dataset", "Seed", "Scenario:Dataset"]
    fits: Dict[str, Dict[str, float]] = {}

    current_terms: List[str] = ["Intercept"]
    fits["Intercept"] = _ols_fit(_concat_blocks(blocks, current_terms), y)

    rows: List[Dict[str, object]] = []
    for term in order[1:]:
        prev_fit = fits[current_terms[-1] if len(current_terms) == 1 else "+".join(current_terms)]
        current_terms.append(term)
        key = "+".join(current_terms)
        current_fit = _ols_fit(_concat_blocks(blocks, current_terms), y)
        fits[key] = current_fit

        ss_term = prev_fit["rss"] - current_fit["rss"]
        df_term = prev_fit["rank"] - current_fit["rank"]
        df_term = abs(df_term)
        rows.append(
            {
                "Term": term,
                "df": int(df_term),
                "SS": float(ss_term),
            }
        )

    full_fit = fits["+".join(order)]
    mse = full_fit["rss"] / full_fit["df_resid"] if full_fit["df_resid"] > 0 else np.nan
    for row in rows:
        row["MS"] = float(row["SS"] / row["df"]) if row["df"] > 0 else np.nan
        row["F"] = float(row["MS"] / mse) if mse and not np.isnan(mse) and row["df"] > 0 else np.nan
        row["p_value"] = (
            float(stats.f.sf(row["F"], row["df"], full_fit["df_resid"]))
            if row["df"] > 0 and full_fit["df_resid"] > 0 and not np.isnan(row["F"])
            else np.nan
        )
        row["partial_eta_sq"] = (
            float(row["SS"] / (row["SS"] + full_fit["rss"]))
            if (row["SS"] + full_fit["rss"]) > 0
            else np.nan
        )

    total_ss = float(np.sum((y - np.mean(y)) ** 2))
    anova_df = pd.DataFrame(rows)
    residual_row = pd.DataFrame(
        [
            {
                "Term": "Residual",
                "df": int(full_fit["df_resid"]),
                "SS": float(full_fit["rss"]),
                "MS": mse,
                "F": np.nan,
                "p_value": np.nan,
                "partial_eta_sq": np.nan,
            }
        ]
    )
    total_row = pd.DataFrame(
        [
            {
                "Term": "Total",
                "df": int(full_fit["n_obs"] - 1),
                "SS": total_ss,
                "MS": np.nan,
                "F": np.nan,
                "p_value": np.nan,
                "partial_eta_sq": np.nan,
            }
        ]
    )
    anova_df = pd.concat([anova_df, residual_row, total_row], ignore_index=True)
    model_info = {
        "metric": metric,
        "n_obs": full_fit["n_obs"],
        "r2": full_fit["r2"],
        "rss": full_fit["rss"],
        "mse": mse,
        "df_resid": full_fit["df_resid"],
        "scenario_levels": int(model_df[FACTOR_SCENARIO].nunique()),
        "dataset_levels": int(model_df[FACTOR_DATASET].nunique()),
        "seed_levels": int(model_df[FACTOR_SEED].nunique()),
    }
    return anova_df, model_info


def plot_seedmean_metric_bars(summary_seedmean: pd.DataFrame, figs_dir: Path) -> None:
    df_plot = _scenario_label_cols(summary_seedmean)
    for metric in METRICS:
        mean_col = f"{metric}_mean"
        ci_col = f"{metric}_ci95_half"
        plot_df = df_plot[["label", mean_col, ci_col]].copy().sort_values(mean_col, ascending=True)
        values = plot_df[mean_col].astype(float).to_numpy() * 100.0
        ci_vals = plot_df[ci_col].astype(float).to_numpy() * 100.0
        y = np.arange(len(plot_df))

        plt.figure(figsize=(11, max(4, 0.55 * len(plot_df))))
        bars = plt.barh(y, values, xerr=ci_vals, capsize=4, color=BAR_COLOR, edgecolor=EDGE_COLOR)
        plt.yticks(y, plot_df["label"].tolist(), fontsize=10)
        plt.xlabel(f"{metric} mean over seeds (%)")
        plt.title(f"{R4_NAME} Confirmatory - {metric} by scenario (95% CI)")
        plt.grid(axis="x", linestyle="--", alpha=0.35)
        for bar, ci in zip(bars, ci_vals):
            x = float(bar.get_width())
            yy = bar.get_y() + bar.get_height() / 2
            plt.text(x + max(ci, 0.05) + 0.1, yy, f"{x:.3f}%", va="center", ha="left", fontsize=9)
        plt.tight_layout()
        plt.savefig(figs_dir / f"{R4_NAME}_confirmatory_seedmean_{_metric_slug(metric)}_bars.png", dpi=220, bbox_inches="tight")
        plt.close()


def plot_dataset_heatmaps(summary_by_scenario_dataset: pd.DataFrame, figs_dir: Path) -> None:
    df_plot = _scenario_label_cols(summary_by_scenario_dataset)
    for metric in METRICS:
        mean_col = f"{metric}_mean"
        pivot = (
            df_plot.pivot_table(index="label", columns=FACTOR_DATASET, values=mean_col, aggfunc="mean")
            .sort_index()
        )
        if pivot.empty:
            continue
        mat = pivot.to_numpy(dtype=float) * 100.0
        plt.figure(figsize=(max(9, 1.0 * pivot.shape[1]), max(4, 0.55 * pivot.shape[0])))
        im = plt.imshow(mat, aspect="auto", cmap=HEATMAP_CMAP)
        plt.title(f"{R4_NAME} Confirmatory - {metric} by scenario x dataset (%)")
        plt.xticks(range(pivot.shape[1]), pivot.columns.tolist(), rotation=30, ha="right", fontsize=10)
        plt.yticks(range(pivot.shape[0]), pivot.index.tolist(), fontsize=10)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isnan(mat[i, j]):
                    continue
                plt.text(j, i, f"{mat[i, j]:.2f}%", ha="center", va="center", fontsize=8, fontweight="bold")
        cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=9)
        plt.tight_layout()
        plt.savefig(figs_dir / f"{R4_NAME}_confirmatory_by_dataset_{_metric_slug(metric)}_heatmap.png", dpi=220, bbox_inches="tight")
        plt.close()


def plot_repeatability_heatmaps(summary_seedmean: pd.DataFrame, figs_dir: Path) -> None:
    df_plot = _scenario_label_cols(summary_seedmean).sort_values(f"{METRICS[0]}_mean", ascending=False)
    std_rows = []
    cv_rows = []
    labels = df_plot["label"].tolist()
    for _, row in df_plot.iterrows():
        std_row = {"label": row["label"]}
        cv_row = {"label": row["label"]}
        for metric in METRICS:
            std_row[metric] = float(row[f"{metric}_std"]) * 100.0
            cv_row[metric] = float(row[f"{metric}_cv_pct"])
        std_rows.append(std_row)
        cv_rows.append(cv_row)

    for suffix, rows, title in [
        ("std", std_rows, "seed repeatability std (%)"),
        ("cv", cv_rows, "seed repeatability CV (%)"),
    ]:
        mat_df = pd.DataFrame(rows).set_index("label")
        mat = mat_df.to_numpy(dtype=float)
        plt.figure(figsize=(8, max(4, 0.55 * len(labels))))
        im = plt.imshow(mat, aspect="auto", cmap=HEATMAP_CMAP)
        plt.title(f"{R4_NAME} Confirmatory - {title}")
        plt.xticks(range(mat_df.shape[1]), mat_df.columns.tolist(), rotation=20, ha="right", fontsize=10)
        plt.yticks(range(mat_df.shape[0]), mat_df.index.tolist(), fontsize=10)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isnan(mat[i, j]):
                    continue
                fmt = f"{mat[i, j]:.3f}" if suffix == "std" else f"{mat[i, j]:.2f}"
                plt.text(j, i, fmt, ha="center", va="center", fontsize=8, fontweight="bold")
        cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=9)
        plt.tight_layout()
        plt.savefig(figs_dir / f"{R4_NAME}_confirmatory_repeatability_{suffix}_heatmap.png", dpi=220, bbox_inches="tight")
        plt.close()


def plot_anova_effects(anova_tables: Dict[str, pd.DataFrame], figs_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    axes = axes.flatten()
    for ax, metric in zip(axes, METRICS):
        anova_df = anova_tables[metric]
        plot_df = anova_df[anova_df["Term"].isin(EFFECT_TERMS)].copy()
        if plot_df.empty:
            ax.set_title(metric, fontsize=16, fontweight="bold")
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=13)
            continue

        effect_vals = plot_df["partial_eta_sq"].astype(float).to_numpy()
        p_vals = plot_df["p_value"].astype(float).to_numpy()
        x_pos = np.arange(len(plot_df), dtype=float)
        colors = ["#d7301f" if (not np.isnan(p) and p < 0.05) else "#ffb347" for p in p_vals]
        ax.bar(x_pos, effect_vals, color=colors, edgecolor=EDGE_COLOR)
        ax.set_title(metric, fontsize=16, fontweight="bold")
        ax.set_ylabel("Partial eta^2", fontsize=16)
        ax.set_ylim(0.0, min(1.05, max(0.15, np.nanmax(effect_vals) * 1.1)))
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_df["Term"].tolist())
        ax.tick_params(axis="x", labelsize=13, rotation=20)
        ax.tick_params(axis="y", labelsize=13)
        for x, eta, p in zip(x_pos, effect_vals, p_vals):
            label = "nan" if np.isnan(p) else f"p={p:.2e}"
            ax.text(x, eta + 0.01, label, ha="center", va="bottom", fontsize=13, rotation=90)
    plt.tight_layout()
    plt.savefig(figs_dir / f"{R4_NAME}_confirmatory_anova_effects.png", dpi=300, bbox_inches="tight")
    plt.close()


def write_report(
    out_path: Path,
    df: pd.DataFrame,
    summary_by_scenario: pd.DataFrame,
    summary_by_scenario_seedmean: pd.DataFrame,
    noise_df: pd.DataFrame,
    model_infos: List[Dict[str, float]],
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("R4 Confirmatory Robustness and Repeatability Report\n")
        f.write("=" * 56 + "\n\n")
        f.write(f"Source rows: {len(df)}\n")
        f.write(f"Scenarios: {df[FACTOR_SCENARIO].nunique()}\n")
        f.write(f"Datasets: {df[FACTOR_DATASET].nunique()}\n")
        f.write(f"Seeds: {df[FACTOR_SEED].nunique()}\n\n")

        f.write("Scenario-level summary files provide mean, std, CV and 95% CI.\n")
        f.write("The seedmean file first averages over datasets within each seed,\n")
        f.write("then summarizes repeatability across the 3 seeds.\n")
        f.write("ANOVA/GLM files use the sequential model:\n")
        f.write("metric ~ Scenario + Dataset + Seed + Scenario:Dataset\n\n")

        f.write("Model fit overview\n")
        f.write("-" * 18 + "\n")
        for info in model_infos:
            f.write(
                f"{info['metric']}: n={info['n_obs']}, R2={info['r2']:.6f}, "
                f"df_resid={info['df_resid']}, mse={info['mse']:.8f}\n"
            )

        f.write("\nSeed-noise reference versus 2 percentage points\n")
        f.write("-" * 46 + "\n")
        for _, row in noise_df.iterrows():
            f.write(
                f"{row['Metric']}: mean_std={row['mean_within_cell_std']:.6f}, "
                f"mean_ci95_half={row['mean_ci95_half_width']:.6f}, "
                f"delta(0.02)/mean_std={row['delta_over_mean_std']:.2f}, "
                f"delta_gt_max_std={row['delta_gt_max_std']}\n"
            )

        f.write("\nTop scenario means across all 18 observations\n")
        f.write("-" * 42 + "\n")
        top_cols = [FACTOR_SCENARIO, MODEL_ID_COL]
        metric_cols = [f"{m}_mean" for m in METRICS]
        top_df = summary_by_scenario[top_cols + metric_cols].copy()
        for _, row in top_df.sort_values("mAP50-95_mean", ascending=False).iterrows():
            f.write(
                f"{row[MODEL_ID_COL]} | {row[FACTOR_SCENARIO]} | "
                f"mAP50-95={row['mAP50-95_mean']:.6f} | "
                f"mAP50={row['mAP50_mean']:.6f} | "
                f"Precision={row['Precision_mean']:.6f} | "
                f"Recall={row['Recall_mean']:.6f}\n"
            )

        f.write("\nScenario repeatability over seed-averaged multidomain means\n")
        f.write("-" * 55 + "\n")
        seed_cols = [FACTOR_SCENARIO, MODEL_ID_COL]
        seed_metric_cols = [f"{m}_std" for m in METRICS]
        seed_df = summary_by_scenario_seedmean[seed_cols + seed_metric_cols].copy()
        for _, row in seed_df.sort_values("mAP50-95_std", ascending=True).iterrows():
            f.write(
                f"{row[MODEL_ID_COL]} | {row[FACTOR_SCENARIO]} | "
                f"std(mAP50-95)={row['mAP50-95_std']:.6f} | "
                f"std(mAP50)={row['mAP50_std']:.6f} | "
                f"std(Precision)={row['Precision_std']:.6f} | "
                f"std(Recall)={row['Recall_std']:.6f}\n"
            )


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / FIGS_DIRNAME
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = load_r4_raw(args.input_csv)

    summary_by_scenario = summarize_groups(df, [FACTOR_SCENARIO], alpha=args.alpha)
    summary_by_scenario = summary_by_scenario.sort_values(f"{METRICS[0]}_mean", ascending=False)
    summary_by_scenario.to_csv(
        out_dir / f"{R4_NAME}_confirmatory_summary_by_scenario.csv",
        index=False,
    )

    df_seedmean = build_seedmean_frame(df)
    summary_by_scenario_seedmean = summarize_groups(
        df_seedmean,
        [FACTOR_SCENARIO],
        alpha=args.alpha,
    )
    summary_by_scenario_seedmean = summary_by_scenario_seedmean.sort_values(
        f"{METRICS[0]}_mean",
        ascending=False,
    )
    summary_by_scenario_seedmean.to_csv(
        out_dir / f"{R4_NAME}_confirmatory_summary_by_scenario_seedmean.csv",
        index=False,
    )

    summary_by_scenario_dataset = summarize_groups(
        df,
        [FACTOR_SCENARIO, FACTOR_DATASET],
        alpha=args.alpha,
    )
    summary_by_scenario_dataset.to_csv(
        out_dir / f"{R4_NAME}_confirmatory_summary_by_scenario_dataset.csv",
        index=False,
    )

    long_df = build_long_for_minitab(df)
    long_df.to_csv(
        out_dir / f"{R4_NAME}_confirmatory_long_for_minitab.csv",
        index=False,
    )

    noise_df = seed_noise_profile(summary_by_scenario_dataset)
    noise_df.to_csv(
        out_dir / f"{R4_NAME}_confirmatory_seed_noise_profile.csv",
        index=False,
    )

    model_infos: List[Dict[str, float]] = []
    anova_tables: Dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        anova_df, model_info = sequential_anova(df, metric)
        model_infos.append(model_info)
        anova_tables[metric] = anova_df
        anova_df.to_csv(
            out_dir / f"{R4_NAME}_confirmatory_anova_{_metric_slug(metric)}.csv",
            index=False,
        )

    plot_seedmean_metric_bars(summary_by_scenario_seedmean, figs_dir)
    plot_dataset_heatmaps(summary_by_scenario_dataset, figs_dir)
    plot_repeatability_heatmaps(summary_by_scenario_seedmean, figs_dir)
    plot_anova_effects(anova_tables, figs_dir)

    write_report(
        out_dir / f"{R4_NAME}_confirmatory_report.txt",
        df=df,
        summary_by_scenario=summary_by_scenario,
        summary_by_scenario_seedmean=summary_by_scenario_seedmean,
        noise_df=noise_df,
        model_infos=model_infos,
    )

    logger.info(f"Analisis confirmatorio R4 generado en: {out_dir}")
    logger.info(f"Figuras confirmatorias R4 generadas en: {figs_dir}")
    print(f"Analisis confirmatorio R4 generado en: {out_dir}")
    print(f"Figuras confirmatorias R4 generadas en: {figs_dir}")


if __name__ == "__main__":
    utils.run_with_sqlite_registration(
        script_name="21_r4_confirmatory_stats.py",
        func=main,
        db_path=config.DB_PATH,
        outputs={"default_out_dir": config.EXPORTS_DIR / R4_NAME},
    )
