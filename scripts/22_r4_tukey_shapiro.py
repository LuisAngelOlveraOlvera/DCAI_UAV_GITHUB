"""
22 R4 TUKEY SHAPIRO ANALYSIS (post-DoE stats)
-----------------------------------------------------------------------
Run confirmatory statistical tests by:
1) reading the R4 raw multi-seed results
2) applying Tukey HSD comparisons
3) checking normality with Shapiro-Wilk summaries
-----------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config
import utils


logger = utils.setup_logger(
    "R4_Tukey_Shapiro",
    log_file=str(config.LOGS_DIR / "r4_tukey_shapiro.log"),
)

R4_NAME = "R4_FINAL"
METRICS = ["mAP50-95", "mAP50", "Precision", "Recall"]
MODEL_ID_COL = "ModelID"
SCENARIO_COL = "scenario_canonical"
SEED_COL = "seed"
DATASET_COL = "Dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tukey HSD and Shapiro-Wilk on R4 outputs."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=config.EXPORTS_DIR / R4_NAME / f"{R4_NAME}_raw_multiseed.csv",
        help="Input CSV with one row per Dataset x Scenario x Seed.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=config.EXPORTS_DIR / R4_NAME / f"{R4_NAME}_tukey_shapiro",
        help="Output directory for tables, report, and figures.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mAP50-95",
        choices=METRICS + ["all"],
        help="Metric to analyze. Use 'all' to run every metric.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="seedmean",
        choices=["seedmean", "raw"],
        help="seedmean averages each metric over datasets within each seed; raw uses Dataset x Seed rows directly.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for Shapiro and Tukey.",
    )
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: faltan columnas requeridas: {missing}")


def _metric_slug(metric: str) -> str:
    return metric.lower().replace("-", "_").replace("@", "").replace(".", "_").replace(":", "_")


def _tight_limits(values: np.ndarray, pad_ratio: float = 0.10, min_pad: float = 0.002) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    vmin = float(vals.min())
    vmax = float(vals.max())
    span = vmax - vmin
    pad = max(min_pad, span * pad_ratio)
    if span == 0:
        pad = max(min_pad, abs(vmin) * 0.10 if vmin != 0 else min_pad)
    return vmin - pad, vmax + pad


def _short_group_labels(df_metric: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for group, grp in df_metric.groupby("Group", sort=False):
        mapping[str(group)] = str(grp[MODEL_ID_COL].iloc[0])
    return mapping


def load_raw_multiseed(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_csv}")

    df = pd.read_csv(input_csv)
    _require_columns(
        df,
        [DATASET_COL, MODEL_ID_COL, SCENARIO_COL, SEED_COL, *METRICS],
        "raw_multiseed",
    )
    for col in [SEED_COL, *METRICS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[DATASET_COL] = df[DATASET_COL].astype(str)
    df[MODEL_ID_COL] = df[MODEL_ID_COL].astype(str)
    df[SCENARIO_COL] = df[SCENARIO_COL].astype(str)
    return df


def build_seedmean_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    return (
        df_raw.groupby([SCENARIO_COL, MODEL_ID_COL, SEED_COL], dropna=False)[METRICS]
        .mean()
        .reset_index()
    )


def prepare_analysis_frame(df_raw: pd.DataFrame, metric: str, mode: str) -> pd.DataFrame:
    if mode == "seedmean":
        base = build_seedmean_frame(df_raw)
    elif mode == "raw":
        base = df_raw[[DATASET_COL, MODEL_ID_COL, SCENARIO_COL, SEED_COL, *METRICS]].copy()
    else:
        raise ValueError(f"Modo no soportado: {mode}")

    df = base.copy()
    df["Value"] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["Value"]).copy()
    df["Group"] = df[MODEL_ID_COL].astype(str) + "|" + df[SCENARIO_COL].astype(str)
    return df


def fit_model_and_residuals(df_metric: pd.DataFrame):
    df = df_metric.copy()
    group_means = df.groupby("Group")["Value"].transform("mean")
    residuals = df["Value"].astype(float) - group_means.astype(float)
    return pd.Series(residuals, name="residual")


def run_shapiro(residuals: pd.Series, alpha: float) -> pd.DataFrame:
    vals = pd.to_numeric(residuals, errors="coerce").dropna()
    if vals.shape[0] < 3:
        raise ValueError("Shapiro-Wilk requiere al menos 3 residuos validos.")

    stat, p_value = stats.shapiro(vals)
    verdict = "normal" if p_value > alpha else "not_normal"
    return pd.DataFrame(
        [
            {
                "test": "Shapiro-Wilk",
                "n_residuals": int(vals.shape[0]),
                "statistic": float(stat),
                "p_value": float(p_value),
                "alpha": float(alpha),
                "verdict": verdict,
            }
        ]
    )


def run_kruskal(df_metric: pd.DataFrame, alpha: float) -> pd.DataFrame:
    groups = [grp["Value"].to_numpy(dtype=float) for _, grp in df_metric.groupby("Group")]
    if len(groups) < 2:
        raise ValueError("Kruskal-Wallis requiere al menos 2 grupos.")
    stat, p_value = stats.kruskal(*groups)
    verdict = "different_groups" if p_value < alpha else "no_detected_difference"
    return pd.DataFrame(
        [
            {
                "test": "Kruskal-Wallis",
                "n_groups": int(len(groups)),
                "statistic": float(stat),
                "p_value": float(p_value),
                "alpha": float(alpha),
                "verdict": verdict,
            }
        ]
    )


def run_tukey(df_metric: pd.DataFrame, alpha: float) -> pd.DataFrame:
    grouped = [
        (group, grp["Value"].astype(float).to_numpy())
        for group, grp in df_metric.groupby("Group", sort=True)
    ]
    labels = [group for group, _ in grouped]
    arrays = [values for _, values in grouped]
    tukey = stats.tukey_hsd(*arrays)
    ci = tukey.confidence_interval(confidence_level=1.0 - alpha)

    rows: List[Dict[str, object]] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            p_adj = float(tukey.pvalue[i, j])
            rows.append(
                {
                    "group1": labels[i],
                    "group2": labels[j],
                    "meandiff": float(tukey.statistic[i, j]),
                    "p-adj": p_adj,
                    "lower": float(ci.low[i, j]),
                    "upper": float(ci.high[i, j]),
                    "reject": bool(p_adj < alpha),
                }
            )
    return pd.DataFrame(rows)


def summarize_groups(df_metric: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for group, grp in df_metric.groupby("Group", sort=True):
        rows.append(
            {
                "Group": group,
                "ModelID": grp[MODEL_ID_COL].iloc[0],
                "scenario_canonical": grp[SCENARIO_COL].iloc[0],
                "n_obs": int(len(grp)),
                "n_seeds": int(grp[SEED_COL].nunique()) if SEED_COL in grp.columns else np.nan,
                "mean": float(grp["Value"].mean()),
                "std": float(grp["Value"].std(ddof=1)) if len(grp) > 1 else 0.0,
                "min": float(grp["Value"].min()),
                "max": float(grp["Value"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def plot_residual_diagnostics(residuals: pd.Series, out_png: Path, metric: str, mode: str) -> None:
    vals = pd.to_numeric(residuals, errors="coerce").dropna()
    bins = min(10, max(5, int(np.sqrt(len(vals)))))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    stats.probplot(vals, dist="norm", plot=plt)
    plt.title(f"Q-Q plot de residuos\n{metric} | {mode}")

    plt.subplot(1, 2, 2)
    counts, bin_edges, _ = plt.hist(
        vals,
        bins=bins,
        color="skyblue",
        edgecolor="black",
        alpha=0.75,
    )
    mu = float(vals.mean())
    sigma = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    if sigma > 0:
        x = np.linspace(float(bin_edges[0]), float(bin_edges[-1]), 400)
        bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
        y = stats.norm.pdf(x, loc=mu, scale=sigma) * len(vals) * bin_width
        plt.plot(x, y, color="#d7301f", linewidth=2.0, label="Campana normal")
        plt.legend()
    plt.title(f"Histograma de residuos\n{metric} | {mode}")
    plt.xlabel("Residuo")
    plt.ylabel("Frecuencia")

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_group_boxplot(df_metric: pd.DataFrame, out_png: Path, metric: str, mode: str) -> None:
    order = (
        df_metric.groupby("Group")["Value"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    data = [df_metric.loc[df_metric["Group"] == group, "Value"].astype(float).to_numpy() for group in order]

    plt.figure(figsize=(12, max(5, 0.45 * len(order))))
    plt.boxplot(data, vert=False, tick_labels=order, patch_artist=True)
    plt.title(f"Distribucion por grupo\n{metric} | {mode}")
    plt.xlabel(metric)
    plt.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_tukey(df_metric: pd.DataFrame, alpha: float, out_png: Path, metric: str, mode: str) -> None:
    tukey_df = run_tukey(df_metric, alpha=alpha).copy()
    if tukey_df.empty:
        return

    tukey_df["label"] = tukey_df["group1"].astype(str) + " vs " + tukey_df["group2"].astype(str)
    tukey_df = tukey_df.sort_values("meandiff", ascending=True).reset_index(drop=True)
    y = np.arange(len(tukey_df))
    colors = ["#d7301f" if bool(v) else "#ffb347" for v in tukey_df["reject"]]
    centers = tukey_df["meandiff"].astype(float).to_numpy()
    lower_err = centers - tukey_df["lower"].astype(float).to_numpy()
    upper_err = tukey_df["upper"].astype(float).to_numpy() - centers

    plt.figure(figsize=(13, max(5, 0.42 * len(tukey_df))))
    plt.errorbar(
        centers,
        y,
        xerr=np.vstack([lower_err, upper_err]),
        fmt="o",
        ecolor="black",
        color="black",
        capsize=4,
        linestyle="none",
    )
    plt.scatter(centers, y, c=colors, s=50, zorder=3)
    plt.axvline(0.0, color="#8b0000", linestyle="--", linewidth=1.6)
    plt.yticks(y, tukey_df["label"].tolist(), fontsize=9)
    plt.xlabel(f"Diferencia de medias ({metric})")
    plt.title(f"Tukey HSD\n{metric} | {mode}")
    plt.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def _plot_tukey_panel(ax, df_metric: pd.DataFrame, alpha: float, metric: str, mode: str) -> None:
    tukey_df = run_tukey(df_metric, alpha=alpha).copy()
    if tukey_df.empty:
        ax.set_axis_off()
        return

    short_labels = _short_group_labels(df_metric)
    tukey_df["label"] = (
        tukey_df["group1"].astype(str).map(short_labels).fillna(tukey_df["group1"].astype(str))
        + " vs "
        + tukey_df["group2"].astype(str).map(short_labels).fillna(tukey_df["group2"].astype(str))
    )
    tukey_df = tukey_df.sort_values("meandiff", ascending=True).reset_index(drop=True)
    y = np.arange(len(tukey_df))
    centers = tukey_df["meandiff"].astype(float).to_numpy()
    lowers = tukey_df["lower"].astype(float).to_numpy()
    uppers = tukey_df["upper"].astype(float).to_numpy()
    lower_err = centers - lowers
    upper_err = uppers - centers
    colors = ["#d7301f" if bool(v) else "#ffb347" for v in tukey_df["reject"]]

    ax.errorbar(
        centers,
        y,
        xerr=np.vstack([lower_err, upper_err]),
        fmt="o",
        ecolor="black",
        color="black",
        capsize=4,
        linestyle="none",
        zorder=2,
    )
    ax.scatter(centers, y, c=colors, s=42, zorder=3)
    ax.axvline(0.0, color="#8b0000", linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(tukey_df["label"].tolist(), fontsize=11)
    ax.set_xlabel(f"Diferencia de medias ({metric})", fontsize=12)
    ax.set_title(f"Tukey HSD\n{metric} | {mode}", fontsize=14, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.30)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xlim(*_tight_limits(np.concatenate([lowers, uppers, np.array([0.0])]), pad_ratio=0.08, min_pad=0.001))


def _plot_group_boxplot_panel(ax, df_metric: pd.DataFrame, metric: str, mode: str) -> None:
    short_labels = _short_group_labels(df_metric)
    order = (
        df_metric.groupby("Group")["Value"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    data = [df_metric.loc[df_metric["Group"] == group, "Value"].astype(float).to_numpy() for group in order]
    tick_labels = [short_labels.get(str(group), str(group)) for group in order]
    ax.boxplot(data, vert=False, tick_labels=tick_labels, patch_artist=True)
    ax.set_title(f"Distribucion por grupo\n{metric} | {mode}", fontsize=14, fontweight="bold")
    ax.set_xlabel(metric, fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.30)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    flat_values = np.concatenate([arr for arr in data if arr.size > 0]) if data else np.array([0.0])
    ax.set_xlim(*_tight_limits(flat_values, pad_ratio=0.10, min_pad=0.001))


def plot_metric_batches(df_raw: pd.DataFrame, mode: str, alpha: float, out_dir: Path) -> None:
    fig_tukey, axes_tukey = plt.subplots(2, 2, figsize=(19, 13))
    axes_tukey = axes_tukey.flatten()
    fig_box, axes_box = plt.subplots(2, 2, figsize=(19, 13))
    axes_box = axes_box.flatten()

    for ax_t, ax_b, metric in zip(axes_tukey, axes_box, METRICS):
        df_metric = prepare_analysis_frame(df_raw, metric=metric, mode=mode)
        if df_metric.empty:
            ax_t.set_axis_off()
            ax_b.set_axis_off()
            continue
        _plot_tukey_panel(ax_t, df_metric, alpha=alpha, metric=metric, mode=mode)
        _plot_group_boxplot_panel(ax_b, df_metric, metric=metric, mode=mode)

    fig_tukey.tight_layout()
    fig_tukey.savefig(out_dir / f"tukey_batch4_{mode}.png", dpi=260, bbox_inches="tight")
    plt.close(fig_tukey)

    fig_box.tight_layout()
    fig_box.savefig(out_dir / f"group_boxplot_batch4_{mode}.png", dpi=260, bbox_inches="tight")
    plt.close(fig_box)


def write_report(
    out_txt: Path,
    metric: str,
    mode: str,
    source_csv: Path,
    summary_df: pd.DataFrame,
    shapiro_df: pd.DataFrame,
    kruskal_df: pd.DataFrame,
    tukey_df: pd.DataFrame,
) -> None:
    shapiro_row = shapiro_df.iloc[0]
    kruskal_row = kruskal_df.iloc[0]
    significant_pairs = tukey_df[tukey_df["reject"].astype(str).str.lower() == "true"].copy()

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("R4 Tukey + Shapiro-Wilk report\n")
        f.write("=" * 34 + "\n\n")
        f.write(f"Metric: {metric}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Source CSV: {source_csv}\n")
        f.write("Recommended source choice: raw_multiseed -> seedmean by scenario/seed.\n")
        f.write("This mirrors the confirmatory aggregation in 21_r4_confirmatory_stats.py\n")
        f.write("and is preferable to moment-only outputs from 20_plot_r4_seed_moments.py.\n\n")

        f.write("Shapiro-Wilk\n")
        f.write("-" * 13 + "\n")
        f.write(
            f"statistic={float(shapiro_row['statistic']):.6f}, "
            f"p_value={float(shapiro_row['p_value']):.6f}, "
            f"verdict={shapiro_row['verdict']}\n\n"
        )

        f.write("Kruskal-Wallis\n")
        f.write("-" * 15 + "\n")
        f.write(
            f"statistic={float(kruskal_row['statistic']):.6f}, "
            f"p_value={float(kruskal_row['p_value']):.6f}, "
            f"verdict={kruskal_row['verdict']}\n\n"
        )

        f.write("Group summary\n")
        f.write("-" * 13 + "\n")
        for _, row in summary_df.iterrows():
            f.write(
                f"{row['Group']}: n={int(row['n_obs'])}, mean={row['mean']:.6f}, "
                f"std={row['std']:.6f}, min={row['min']:.6f}, max={row['max']:.6f}\n"
            )

        f.write("\nSignificant Tukey pairs (reject=True)\n")
        f.write("-" * 36 + "\n")
        if significant_pairs.empty:
            f.write("No significant pairwise differences detected.\n")
        else:
            for _, row in significant_pairs.iterrows():
                f.write(
                    f"{row['group1']} vs {row['group2']}: "
                    f"meandiff={float(row['meandiff']):.6f}, "
                    f"p-adj={float(row['p-adj']):.6f}, reject={row['reject']}\n"
                )


def analyze_metric(
    df_raw: pd.DataFrame,
    metric: str,
    mode: str,
    alpha: float,
    out_dir: Path,
    source_csv: Path,
) -> None:
    metric_slug = _metric_slug(metric)
    df_metric = prepare_analysis_frame(df_raw, metric=metric, mode=mode)
    if df_metric.empty:
        raise ValueError(f"No hay observaciones validas para {metric} en modo {mode}.")

    metric_dir = out_dir / f"{metric_slug}_{mode}"
    metric_dir.mkdir(parents=True, exist_ok=True)

    df_metric.to_csv(metric_dir / f"analysis_input_{metric_slug}.csv", index=False)

    summary_df = summarize_groups(df_metric)
    summary_df.to_csv(metric_dir / f"group_summary_{metric_slug}.csv", index=False)

    residuals = fit_model_and_residuals(df_metric)
    shapiro_df = run_shapiro(residuals, alpha=alpha)
    shapiro_df.to_csv(metric_dir / f"shapiro_wilk_{metric_slug}.csv", index=False)

    kruskal_df = run_kruskal(df_metric, alpha=alpha)
    kruskal_df.to_csv(metric_dir / f"kruskal_wallis_{metric_slug}.csv", index=False)

    tukey_df = run_tukey(df_metric, alpha=alpha)
    tukey_df.to_csv(metric_dir / f"tukey_hsd_{metric_slug}.csv", index=False)

    plot_residual_diagnostics(
        residuals,
        metric_dir / f"residual_diagnostics_{metric_slug}.png",
        metric=metric,
        mode=mode,
    )
    plot_group_boxplot(
        df_metric,
        metric_dir / f"group_boxplot_{metric_slug}.png",
        metric=metric,
        mode=mode,
    )
    plot_tukey(
        df_metric,
        alpha=alpha,
        out_png=metric_dir / f"tukey_plot_{metric_slug}.png",
        metric=metric,
        mode=mode,
    )

    write_report(
        metric_dir / f"report_{metric_slug}.txt",
        metric=metric,
        mode=mode,
        source_csv=source_csv,
        summary_df=summary_df,
        shapiro_df=shapiro_df,
        kruskal_df=kruskal_df,
        tukey_df=tukey_df,
    )

    logger.info(f"Analisis completado para {metric} ({mode}) en: {metric_dir}")
    print(f"Analisis completado para {metric} ({mode}) en: {metric_dir}")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_multiseed(args.input_csv)
    metrics_to_run = METRICS if args.metric == "all" else [args.metric]

    for metric in metrics_to_run:
        analyze_metric(
            df_raw=df_raw,
            metric=metric,
            mode=args.mode,
            alpha=args.alpha,
            out_dir=out_dir,
            source_csv=args.input_csv,
        )

    if args.metric == "all":
        plot_metric_batches(
            df_raw=df_raw,
            mode=args.mode,
            alpha=args.alpha,
            out_dir=out_dir,
        )
        logger.info(f"Figuras batch-4 generadas en: {out_dir}")
        print(f"Figuras batch-4 generadas en: {out_dir}")


if __name__ == "__main__":
    utils.run_with_sqlite_registration(
        script_name="22_r4_tukey_shapiro.py",
        func=main,
        db_path=config.DB_PATH,
        outputs={"default_out_dir": config.EXPORTS_DIR / R4_NAME / f"{R4_NAME}_tukey_shapiro"},
    )
