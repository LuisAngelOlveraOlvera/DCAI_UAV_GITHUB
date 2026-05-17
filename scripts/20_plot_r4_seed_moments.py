"""
20 R4 SEED MOMENT PLOTS (post-DoE analysis)
-----------------------------------------------------------------------
Visualize R4 seed-moment exports by:
1) loading per-dataset and global CSV summaries
2) plotting scenario behavior across moments
3) saving comparative figures for interpretation
-----------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import config
import utils


logger = utils.setup_logger(
    "R4_Seed_Moments_Plot",
    log_file=str(config.LOGS_DIR / "r4_seed_moments_plot.log"),
)

R4_NAME = "R4_FINAL"
METRICS = ["mAP50-95", "mAP50", "Precision", "Recall"]
MOMENT_STATS = ["mean", "std", "skew", "kurtosis"]
HEATMAP_COLORS = ["#ffffff", "#fff3b0", "#ffb347", "#d7301f"]
HEATMAP_CMAP = LinearSegmentedColormap.from_list("white_yellow_orange_red_moments", HEATMAP_COLORS)
PERCENT_SCALE = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot R4 seed-moment CSV artifacts.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=config.EXPORTS_DIR / R4_NAME,
        help="Directory containing R4_FINAL_seed_moments_by_dataset.csv and R4_FINAL_seed_moments_global.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=(config.EXPORTS_DIR / R4_NAME / f"{R4_NAME}_moments_figs"),
        help="Output directory for figures",
    )
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: faltan columnas: {miss}")


def _label_col(df: pd.DataFrame) -> pd.Series:
    return df["ModelID"].astype(str) + "|" + df["scenario_canonical"].astype(str)


def plot_global_mean_std(df_global: pd.DataFrame, out_dir: Path) -> None:
    _require_columns(df_global, ["ModelID", "scenario_canonical"], "global")
    df = df_global.copy()
    df["label"] = _label_col(df)

    for metric in METRICS:
        mean_col = f"{metric}_mean"
        _require_columns(df, [mean_col], "global")

        d = df[["label", mean_col]].copy().sort_values(mean_col, ascending=True)
        y = np.arange(len(d))
        mean_pct = d[mean_col].astype(float).values * PERCENT_SCALE

        plt.figure(figsize=(11, max(4, 0.5 * len(d))))
        bars = plt.barh(y, mean_pct, color="#ffb347", edgecolor="#d7301f")
        plt.yticks(y, d["label"].tolist(), fontsize=10)
        plt.xlabel(f"{metric} (%)")
        plt.title(f"{R4_NAME} - Global {metric}")
        plt.grid(axis="x", linestyle="--", alpha=0.35)
        for bar in bars:
            x = float(bar.get_width())
            yy = bar.get_y() + bar.get_height() / 2
            plt.text(x + 0.2, yy, f"{x:.3f}%", va="center", ha="left", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_dir / f"{R4_NAME}_global_{metric}_mean_std.png", dpi=220, bbox_inches="tight")
        plt.close()


def plot_global_moment_heatmap(df_global: pd.DataFrame, out_dir: Path) -> None:
    _require_columns(df_global, ["ModelID", "scenario_canonical"], "global")
    df = df_global.copy()
    df["label"] = _label_col(df)

    rows = []
    for _, r in df.iterrows():
        row = {"label": r["label"]}
        for metric in METRICS:
            for stat in MOMENT_STATS:
                c = f"{metric}_{stat}"
                row[f"{metric}|{stat}"] = float(r[c]) if c in df.columns else np.nan
        rows.append(row)

    mat_df = pd.DataFrame(rows).set_index("label")
    mat = mat_df.to_numpy(dtype=float)

    plt.figure(figsize=(max(11, 0.4 * mat.shape[1]), max(4, 0.5 * mat.shape[0])))
    im = plt.imshow(mat, aspect="auto", cmap=HEATMAP_CMAP)
    plt.title(f"{R4_NAME} - Global Seed Moments Heatmap")
    plt.xticks(range(mat_df.shape[1]), mat_df.columns.tolist(), rotation=45, ha="right", fontsize=9)
    plt.yticks(range(mat_df.shape[0]), mat_df.index.tolist(), fontsize=10)
    cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / f"{R4_NAME}_global_seed_moments_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_dataset_metric_heatmaps(df_by_ds: pd.DataFrame, out_dir: Path) -> None:
    _require_columns(df_by_ds, ["ModelID", "scenario_canonical", "Dataset"], "by_dataset")
    df = df_by_ds.copy()
    df["label"] = _label_col(df)

    for metric in METRICS:
        mean_col = f"{metric}_mean"
        _require_columns(df, [mean_col], "by_dataset")
        pivot = (
            df.pivot_table(index="label", columns="Dataset", values=mean_col, aggfunc="mean")
            .sort_index()
            .copy()
        )
        if pivot.empty:
            continue
        mat = pivot.to_numpy(dtype=float) * PERCENT_SCALE

        plt.figure(figsize=(max(9, 0.8 * pivot.shape[1]), max(4, 0.5 * pivot.shape[0])))
        im = plt.imshow(mat, aspect="auto", cmap=HEATMAP_CMAP)
        plt.title(f"{R4_NAME} - {metric} mean by dataset (%)")
        plt.xticks(range(pivot.shape[1]), pivot.columns.tolist(), rotation=30, ha="right", fontsize=10)
        plt.yticks(range(pivot.shape[0]), pivot.index.tolist(), fontsize=10)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isnan(mat[i, j]):
                    continue
                plt.text(j, i, f"{mat[i, j]:.3f}%", ha="center", va="center", fontsize=9, fontweight="bold")
        cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=10)
        plt.tight_layout()
        plt.savefig(out_dir / f"{R4_NAME}_by_dataset_{metric}_mean_heatmap.png", dpi=220, bbox_inches="tight")
        plt.close()


def main() -> None:
    args = parse_args()
    in_dir = args.input_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    by_ds_csv = in_dir / f"{R4_NAME}_seed_moments_by_dataset.csv"
    global_csv = in_dir / f"{R4_NAME}_seed_moments_global.csv"

    if not by_ds_csv.exists():
        raise FileNotFoundError(f"No existe: {by_ds_csv}")
    if not global_csv.exists():
        raise FileNotFoundError(f"No existe: {global_csv}")

    df_by_ds = pd.read_csv(by_ds_csv)
    df_global = pd.read_csv(global_csv)

    plot_global_mean_std(df_global, out_dir)
    plot_global_moment_heatmap(df_global, out_dir)
    plot_dataset_metric_heatmaps(df_by_ds, out_dir)

    logger.info(f"Plots generados en: {out_dir}")
    print(f"Plots generados en: {out_dir}")


if __name__ == "__main__":
    utils.run_with_sqlite_registration(
        script_name="20_plot_r4_seed_moments.py",
        func=main,
        db_path=config.DB_PATH,
        outputs={"default_out_dir": config.EXPORTS_DIR / R4_NAME / f"{R4_NAME}_moments_figs"},
    )
