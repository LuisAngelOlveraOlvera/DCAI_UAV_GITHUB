"""
17 REDUCED DOE PIPELINE (hierarchical rounds)
-----------------------------------------------------------------------
Run the reduced hierarchical DoE process by:
1) executing experimental rounds R0 to R3
2) applying post-R3 gating rules
3) producing robust global scoring for candidate selection
-----------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import config
import utils


logger = utils.setup_logger(
    "DOE_Reduced",
    log_file=str(config.LOGS_DIR / "doe_reduced_pipeline.log"),
)


# Reduced hierarchical DoE mapping (as requested in DOE.md)
DOE_PLAN: Dict[str, List[str]] = {
    "R0": ["Z_COCO", "B_Raw_0"],
    "R1": ["H_Raw_0", "H_pH_0", "H_JR_0", "H_pH_JR_0", "H_pH_JS_0"],
    "R2": ["H_pH_JR_15", "H_pH_JR_40", "H_pH_JS_40"],
    "R3": ["P_Raw_0", "P_pH_JR_0", "P_pH_JR_15", "P_pH_JR_40"],
}

# Expected canonical order (stable export/order for reports)
SCENARIO_ORDER = DOE_PLAN["R0"] + DOE_PLAN["R1"] + DOE_PLAN["R2"] + DOE_PLAN["R3"]
SCENARIO_TO_ROUND = {s: r for r, scenarios in DOE_PLAN.items() for s in scenarios}

# N-index to canonical scenario mapping used in current training/eval naming
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

METRICS = ["mAP50-95", "mAP50", "Precision", "Recall"]
METRIC_LABELS = {
    "mAP50-95": "mAP@0.5:0.95",
    "mAP50": "mAP@0.5",
    "Precision": "Precision",
    "Recall": "Recall",
}

FIGS_ROOT = config.EXPORTS_DIR / "doe_reduced_figs"
HEATMAP_COLORS = ["#ffffff", "#fff3b0", "#ffb347", "#d7301f"]
ABS_CMAP = LinearSegmentedColormap.from_list("white_yellow_orange_red_abs", HEATMAP_COLORS)
DELTA_CMAP = LinearSegmentedColormap.from_list("white_yellow_orange_red_delta", HEATMAP_COLORS)
DELTA_SCALE = 100.0
PERCENT_SCALE = 100.0
OUTPUT_ROOT = config.EXPORTS_DIR / "doe_reduced"

# MCDM fixed constants (do not change values)
ALPHA = 0.50
W_MAP5095 = 0.50
W_MAP50 = 0.20
W_PREC = 0.15
W_REC = 0.15
GAMMA = 1.00
GATING_REFERENCE = "P_Raw_0"



def normalize_scenario(raw_name: str) -> str:
    """
    Normalize raw scenario names to canonical DOE labels.

    Examples:
    - N2_B_Raw_0_yolo11n_e100 -> B_Raw_0
    - N1_YOLO11n -> Z_COCO
    - H_pH_JR_15 -> H_pH_JR_15
    """
    raw_name = str(raw_name).strip()

    if raw_name in SCENARIO_TO_ROUND:
        return raw_name

    m = re.match(r"^(N\d+)", raw_name)
    if m:
        n_id = m.group(1)
        mapped = N_TO_CANONICAL.get(n_id)
        if mapped:
            return mapped

    for canonical in SCENARIO_ORDER:
        if canonical in raw_name:
            return canonical

    return raw_name


def load_master_results() -> pd.DataFrame:
    """
    Load master DoE results.

    Priority:
    1) exports/thesis_results_all_datasets.csv
    2) concatenate exports/results_*.csv
    """
    master_csv = config.EXPORTS_DIR / "thesis_results_all_datasets.csv"
    if master_csv.exists():
        df = pd.read_csv(master_csv)
        logger.info(f"Master CSV loaded: {master_csv}")
        return df

    partials = sorted(config.EXPORTS_DIR.glob("results_*.csv"))
    if not partials:
        raise FileNotFoundError(
            "No se encontró thesis_results_all_datasets.csv ni results_*.csv en exports."
        )

    frames = [pd.read_csv(p) for p in partials]
    df = pd.concat(frames, ignore_index=True)
    out_master = config.EXPORTS_DIR / "thesis_results_all_datasets.csv"
    df.to_csv(out_master, index=False)
    logger.info(f"Master CSV generado desde parciales: {out_master}")
    return df


def validate_master(df: pd.DataFrame) -> None:
    required = {"Dataset", "Escenario", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en master CSV: {sorted(missing)}")


def check_round_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check scenario completeness per round and dataset.
    """
    rows = []
    datasets = sorted(df["Dataset"].dropna().unique().tolist())
    for round_name, expected in DOE_PLAN.items():
        expected_set = set(expected)

        present_global = set(df.loc[df["scenario_canonical"].isin(expected), "scenario_canonical"])
        missing_global = sorted(expected_set - present_global)
        rows.append(
            {
                "Scope": "GLOBAL",
                "Round": round_name,
                "Expected": len(expected),
                "Present": len(present_global),
                "Missing": ", ".join(missing_global),
            }
        )

        for ds in datasets:
            df_ds = df[df["Dataset"] == ds]
            present_ds = set(
                df_ds.loc[df_ds["scenario_canonical"].isin(expected), "scenario_canonical"]
            )
            missing_ds = sorted(expected_set - present_ds)
            rows.append(
                {
                    "Scope": ds,
                    "Round": round_name,
                    "Expected": len(expected),
                    "Present": len(present_ds),
                    "Missing": ", ".join(missing_ds),
                }
            )

    return pd.DataFrame(rows)


def build_round_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean metrics per scenario across datasets + round annotation.
    """
    summary = (
        df.groupby("scenario_canonical", as_index=False)[METRICS]
        .mean(numeric_only=True)
        .copy()
    )
    summary["Round"] = summary["scenario_canonical"].map(SCENARIO_TO_ROUND)
    summary = summary[summary["scenario_canonical"].isin(SCENARIO_ORDER)]
    summary["order"] = summary["scenario_canonical"].map({s: i for i, s in enumerate(SCENARIO_ORDER)})
    summary = summary.sort_values("order").drop(columns=["order"])
    return summary


def build_summary_for_round(df_round: pd.DataFrame) -> pd.DataFrame:
    """
    Summary with mean/min columns per metric for a specific round.
    """
    summary = df_round.groupby("scenario_canonical")[METRICS].agg(["mean", "min"])
    summary.columns = [
        f"{metric}_{'Promedio' if agg == 'mean' else 'PeorCaso'}"
        for metric, agg in summary.columns
    ]
    col_order = []
    for m in METRICS:
        col_order += [f"{m}_Promedio", f"{m}_PeorCaso"]
    summary = summary[col_order].reset_index()
    order_map = {s: i for i, s in enumerate(SCENARIO_ORDER)}
    summary["order"] = summary["scenario_canonical"].map(order_map)
    return summary.sort_values("order").drop(columns=["order"])


def build_delta_vs_baseline(df: pd.DataFrame, baseline: str = "B_Raw_0") -> pd.DataFrame:
    """
    Per-dataset deltas against baseline scenario.
    """
    rows = []
    for ds in sorted(df["Dataset"].dropna().unique().tolist()):
        df_ds = df[df["Dataset"] == ds]
        b = df_ds[df_ds["scenario_canonical"] == baseline]
        if b.empty:
            logger.warning(f"{ds}: baseline {baseline} no disponible para deltas.")
            continue
        b_vals = b.iloc[0][METRICS]

        for _, r in df_ds.iterrows():
            s = r["scenario_canonical"]
            if s not in SCENARIO_ORDER:
                continue
            out = {"Dataset": ds, "scenario_canonical": s}
            for m in METRICS:
                out[f"delta_{m}"] = float(r[m]) - float(b_vals[m])
            rows.append(out)
    return pd.DataFrame(rows)


def plot_dataset_bars(df_round: pd.DataFrame, round_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset in sorted(df_round["Dataset"].dropna().unique().tolist()):
        df_ds = df_round[df_round["Dataset"] == dataset].copy()
        if df_ds.empty:
            continue
        df_ds = df_ds.sort_values("mAP50-95", ascending=True)

        scenarios = df_ds["scenario_canonical"].tolist()
        x = np.arange(len(scenarios))
        bar_w = 0.18

        plt.figure(figsize=(max(10, 1.2 * len(scenarios)), 6))
        for i, metric in enumerate(METRICS):
            values = df_ds[metric].astype(float).values * PERCENT_SCALE
            bars = plt.bar(x + i * bar_w, values, bar_w, label=METRIC_LABELS[metric])
            for bar in bars:
                h = float(bar.get_height())
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.8,
                    f"{h:.3f}%",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    rotation=90,
                )

        plt.title(f"{round_name} - {dataset} (ordenado por mAP50-95)")
        plt.xlabel("Escenario")
        plt.ylabel("Score (%)")
        plt.ylim(0, 105)
        plt.xticks(x + bar_w * (len(METRICS) - 1) / 2, scenarios, rotation=30, ha="right", fontsize=11)
        plt.yticks(fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{round_name}_bars_{dataset}.png", dpi=200, bbox_inches="tight")
        plt.close()


def plot_abs_summary_heatmap(df_summary: pd.DataFrame, round_name: str, out_png: Path) -> None:
    cols = [c for c in df_summary.columns if c != "scenario_canonical"]
    if not cols:
        return
    mat = df_summary[cols].to_numpy(dtype=float) * PERCENT_SCALE

    row_labels = df_summary["scenario_canonical"].tolist()
    fig_w = max(10, 0.45 * len(cols))
    fig_h = max(4, 0.45 * len(row_labels))

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat, aspect="auto", cmap=ABS_CMAP)
    plt.title(f"{round_name} - Heatmap Resumen (Absoluto, %)")
    plt.xticks(range(len(cols)), cols, rotation=30, ha="right", fontsize=11)
    plt.yticks(range(len(row_labels)), row_labels, fontsize=11)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f"{mat[i, j]:.3f}%", ha="center", va="center", fontsize=12, fontweight="bold")

    cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def delta_matrix_from_series(values: pd.Series) -> pd.DataFrame:
    vals = values.astype(float).values
    diff = (vals[:, None] - vals[None, :]) * DELTA_SCALE
    return pd.DataFrame(diff, index=values.index, columns=values.index)


def plot_delta_heatmap(df_delta: pd.DataFrame, title: str, out_png: Path) -> None:
    mat = df_delta.to_numpy(dtype=float)
    labels = df_delta.index.tolist()

    fig_w = max(6, 0.6 * len(labels))
    fig_h = max(5, 0.6 * len(labels))

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat, aspect="auto", cmap=DELTA_CMAP)
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=11)
    plt.yticks(range(len(labels)), labels, fontsize=11)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(
                j,
                i,
                f"{mat[i, j]:+.3f}%",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )

    cbar = plt.colorbar(im, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def generate_deltas_for_round(df_round: pd.DataFrame, round_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios_order = [
        s for s in SCENARIO_ORDER if s in set(df_round["scenario_canonical"].unique().tolist())
    ]

    for metric in METRICS:
        for dataset in sorted(df_round["Dataset"].dropna().unique().tolist()):
            df_ds = df_round[df_round["Dataset"] == dataset].copy()
            s = df_ds.set_index("scenario_canonical")[metric].reindex(scenarios_order)
            if s.isna().any():
                continue
            df_delta = delta_matrix_from_series(s)
            csv_path = out_dir / f"{round_name}_deltas_{metric}_{dataset}.csv"
            png_path = out_dir / f"{round_name}_deltas_{metric}_{dataset}.png"
            df_delta.to_csv(csv_path)
            plot_delta_heatmap(
                df_delta,
                f"{round_name} - Deltas (Mi-Mj)*{DELTA_SCALE:g} - {metric} - {dataset}",
                png_path,
            )


def plot_post_r3_ranking(ranking: pd.DataFrame, out_png: Path) -> None:
    df = ranking[ranking["passed_gating"]].copy()
    if df.empty:
        return
    df = df.sort_values("S_global", ascending=True)
    df["S_global_pct"] = df["S_global"].astype(float) * PERCENT_SCALE
    plt.figure(figsize=(10, max(4, 0.5 * len(df))))
    bars = plt.barh(df["scenario_canonical"], df["S_global_pct"])
    plt.xlabel("S_global (%)")
    plt.ylabel("Escenario")
    plt.title("Post-R3 Ranking (solo escenarios que pasan gating)")
    plt.grid(axis="x", linestyle="--", alpha=0.35)
    for bar in bars:
        w = float(bar.get_width())
        y = bar.get_y() + bar.get_height() / 2
        plt.text(w + 0.2, y, f"{w:.3f}%", va="center", ha="left", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_post_r3_gating_threshold(ranking: pd.DataFrame, out_png: Path) -> None:
    """
    Shows Recall_min per scenario with explicit gating threshold tau_R.
    """
    if ranking.empty or "Recall_min" not in ranking.columns:
        return
    df = ranking.copy().sort_values("Recall_min", ascending=True)
    df["Recall_min_pct"] = df["Recall_min"].astype(float) * PERCENT_SCALE
    tau_r = float(df["Recall_gate_threshold"].iloc[0]) if "Recall_gate_threshold" in df.columns else None
    tau_r_pct = tau_r * PERCENT_SCALE if tau_r is not None else None

    colors = ["tab:green" if bool(v) else "tab:red" for v in df["passed_gating"]]
    plt.figure(figsize=(12, max(5, 0.45 * len(df))))
    bars = plt.barh(df["scenario_canonical"], df["Recall_min_pct"], color=colors)

    if tau_r_pct is not None:
        plt.axvline(tau_r_pct, color="black", linestyle="--", linewidth=1.8, label=f"tau_R={tau_r_pct:.3f}%")
        plt.legend(loc="lower right")

    plt.xlabel("Recall_min (%)")
    plt.ylabel("Escenario")
    plt.title("Post-R3 Gating: Recall_min vs tau_R (target a superar)")
    plt.grid(axis="x", linestyle="--", alpha=0.35)

    for bar in bars:
        w = float(bar.get_width())
        y = bar.get_y() + bar.get_height() / 2
        plt.text(w + 0.2, y, f"{w:.3f}%", va="center", ha="left", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def apply_gating_and_scoring(
    df: pd.DataFrame,
    alpha: float = ALPHA,
    metric_weights: Dict[str, float] | None = {
        "mAP50-95": W_MAP5095,
        "mAP50": W_MAP50,
        "Precision": W_PREC,
        "Recall": W_REC,
    },
    baseline: str = GATING_REFERENCE,
) -> pd.DataFrame:
    """
    Post-R3 robust selection layer.

    Gating:
      Recall_min(i) >= Recall_min(baseline)

    Score by metric:
      S_m(i) = alpha * mean_m(i) + (1-alpha) * min_m(i)

    Global score:
      S_global(i) = sum(w_m * S_m(i))
    """
    if metric_weights is None:
        metric_weights = {
            "mAP50-95": W_MAP5095,
            "mAP50": W_MAP50,
            "Precision": W_PREC,
            "Recall": W_REC,
        }

    # Validate metric weights
    unknown = set(metric_weights) - set(METRICS)
    if unknown:
        raise ValueError(f"metric_weights contiene métricas desconocidas: {sorted(unknown)}")
    if abs(sum(metric_weights.values()) - 1.0) > 1e-6:
        raise ValueError("metric_weights debe sumar 1.0")

    agg_mean = df.groupby("scenario_canonical")[METRICS].mean(numeric_only=True)
    agg_min = df.groupby("scenario_canonical")[METRICS].min(numeric_only=True)

    if baseline not in agg_min.index:
        logger.warning(
            "Post-R3 ranking skipped: baseline "
            f"'{baseline}' is missing from the results used for gating."
        )
        logger.warning(
            "Expected at least one row for the baseline scenario in order to "
            "compute Recall_min gating and global ranking."
        )
        available = sorted(agg_min.index.tolist())
        logger.warning(f"Available scenarios for ranking: {available}")
        return pd.DataFrame()

    tau_r = float(agg_min.loc[baseline, "Recall"])

    rows = []
    for scenario in SCENARIO_ORDER:
        if scenario not in agg_mean.index or scenario not in agg_min.index:
            continue

        row = {
            "scenario_canonical": scenario,
            "Round": SCENARIO_TO_ROUND.get(scenario, ""),
        }

        for m in METRICS:
            row[f"{m}_mean"] = float(agg_mean.loc[scenario, m])
            row[f"{m}_min"] = float(agg_min.loc[scenario, m])
            row[f"S_{m}"] = alpha * row[f"{m}_mean"] + (1.0 - alpha) * row[f"{m}_min"]

        row["Recall_gate_threshold"] = tau_r
        row["passed_gating"] = row["Recall_min"] >= tau_r
        row["S_global"] = sum(metric_weights[m] * row[f"S_{m}"] for m in METRICS)
        rows.append(row)

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking

    # Rank only candidates that pass gating
    ranking["Rank"] = None
    passed = ranking[ranking["passed_gating"]].copy().sort_values("S_global", ascending=False)
    passed["Rank"] = range(1, len(passed) + 1)
    ranking.loc[passed.index, "Rank"] = passed["Rank"]

    ranking = ranking.sort_values(
        by=["passed_gating", "S_global"],
        ascending=[False, False],
    )
    return ranking


def passes_gating(
    recall_min: float,
    reference_recall_min: float,
    gamma: float = GAMMA,
) -> bool:
    """
    Returns True if recall_min >= gamma * reference_recall_min.
    Implements eq. (5) of the paper.
    Universal gating check used identically in R1, R2, and R3.
    Reference is always Recall_min(P_Raw_0) across all rounds.
    """
    return float(recall_min) >= gamma * float(reference_recall_min)


def score_scenario(
    mean_vals: dict[str, float],
    min_vals: dict[str, float],
) -> float:
    """
    Computes S_global(i) using fixed constants ALPHA and W_*.

    For each metric M in {mAP50-95, mAP50, Precision, Recall}:
        S_M(i) = ALPHA * mean_vals[M] + (1 - ALPHA) * min_vals[M]

    S_global(i) = W_MAP5095 * S_mAP50-95
                + W_MAP50   * S_mAP50
                + W_PREC    * S_Precision
                + W_REC     * S_Recall

    Implements eqs. (7) and (8) of the paper.
    This is the ONLY score function used in R1, R2, and R3.
    No round-specific coefficients exist.
    """
    s_map5095 = ALPHA * float(mean_vals["mAP50-95"]) + (1.0 - ALPHA) * float(min_vals["mAP50-95"])
    s_map50 = ALPHA * float(mean_vals["mAP50"]) + (1.0 - ALPHA) * float(min_vals["mAP50"])
    s_prec = ALPHA * float(mean_vals["Precision"]) + (1.0 - ALPHA) * float(min_vals["Precision"])
    s_rec = ALPHA * float(mean_vals["Recall"]) + (1.0 - ALPHA) * float(min_vals["Recall"])

    return (
        W_MAP5095 * s_map5095
        + W_MAP50 * s_map50
        + W_PREC * s_prec
        + W_REC * s_rec
    )


def select_round_survivors(
    df_round: pd.DataFrame,
    round_name: str,
    p_raw_recall_min: float,
    max_survivors: int,
    force_if_empty: bool = False,
) -> tuple[list[str], dict[str, float], bool]:
    """
    Generic survivor selection for a single round (R1, R2, or R3).
    Rounds are fully isolated - this function has no knowledge of other
    rounds and receives no inter-round inputs except p_raw_recall_min,
    which is the universal gating reference computed once from P_Raw_0.

    Step 1 - Compute per-scenario aggregates from df_round.
    Step 2 - Gating with universal threshold against P_Raw_0.
    Step 3 - Score survivors using score_scenario(mean_vals, min_vals).
    Step 4 - Optional forced fallback when force_if_empty=True.
    Step 5 - Return top candidates sorted by S_global desc.
    """
    if df_round.empty:
        logger.warning(f"{round_name} selection received an empty dataframe.")
        return [], {}, False

    rows = []
    for scenario, df_s in df_round.groupby("scenario_canonical"):
        mean_vals = {metric: float(df_s[metric].astype(float).mean()) for metric in METRICS}
        min_vals = {metric: float(df_s[metric].astype(float).min()) for metric in METRICS}
        rows.append(
            {
                "scenario_canonical": scenario,
                "Recall_min": min_vals["Recall"],
                "S_global": score_scenario(mean_vals, min_vals),
            }
        )

    grouped = pd.DataFrame(rows)
    if grouped.empty:
        return [], {}, False

    threshold = GAMMA * float(p_raw_recall_min)
    keep_mask = []
    for row in grouped.itertuples(index=False):
        passed = passes_gating(row.Recall_min, p_raw_recall_min)
        keep_mask.append(passed)
        if not passed:
            logger.warning(
                f"{row.scenario_canonical} failed {round_name} gating: "
                f"Recall_min={row.Recall_min:.4f} < threshold={threshold:.4f}"
            )

    survivors_df = grouped.loc[keep_mask].copy()
    if survivors_df.empty:
        logger.warning(f"{round_name} produced 0 survivors after gating.")
        if not force_if_empty:
            return [], {}, False

        logger.warning(
            f"{round_name} produced 0 survivors after gating. Forcing best by S_global (forced=True)."
        )
        candidates_df = grouped.copy()
        forced = True
    else:
        candidates_df = survivors_df.copy()
        forced = False

    candidates_df = candidates_df.sort_values(
        by=["S_global", "scenario_canonical"],
        ascending=[False, True],
    ).head(max_survivors)

    if candidates_df.empty:
        return [], {}, False

    survivors = candidates_df["scenario_canonical"].tolist()
    scores = dict(zip(candidates_df["scenario_canonical"], candidates_df["S_global"]))
    return survivors, scores, forced


def assemble_r4_finalists(
    round_survivors: dict[str, list[str]],
    r2_forced: bool,
) -> list[str]:
    """
    Assembles the R4 finalist list guaranteeing at least one representative
    per round where possible.

    Slots (in order):
    1. R0 anchor  : always "B_Raw_0". Never include "Z_COCO" as a finalist.
    2. R1 slot    : first element of round_survivors["R1"] (top by S_global).
                    If R1 is empty, log WARNING and skip slot.
    3. R2 slot    : first element of round_survivors["R2"] (top by S_global,
                    possibly forced). If r2_forced is True, log INFO:
                        f"R2 finalist is forced (no scenario passed R2 gating): {scenario}"
                    If R2 is empty, log WARNING and skip slot.
    4. R3 slots   : ALL elements of round_survivors["R3"] (up to max_survivors=3).
                    If R3 is empty, log WARNING and skip slot.

    Deduplication: if any scenario appears in more than one slot, keep only
    its first occurrence. Preserve order R0 -> R1 -> R2 -> R3.

    Log final list at INFO:
        f"R4 finalists assembled ({len(finalists)}): {finalists}"

    Returns:
        finalists: list[str]
    """
    finalists_ordered = ["B_Raw_0"]

    r1_survivors = round_survivors.get("R1", [])
    if r1_survivors:
        finalists_ordered.append(r1_survivors[0])
    else:
        logger.warning("R1 produced 0 survivors; skipping R1 slot.")

    r2_survivors = round_survivors.get("R2", [])
    if r2_survivors:
        r2_finalist = r2_survivors[0]
        finalists_ordered.append(r2_finalist)
        if r2_forced:
            logger.info(f"R2 finalist is forced (no scenario passed R2 gating): {r2_finalist}")
    else:
        logger.warning("R2 produced 0 survivors; skipping R2 slot.")

    r3_survivors = round_survivors.get("R3", [])
    if r3_survivors:
        finalists_ordered.extend(r3_survivors)
    else:
        logger.warning("R3 produced 0 survivors; skipping R3 slot.")

    seen = set()
    finalists = []
    for scenario in finalists_ordered:
        if scenario in seen:
            continue
        finalists.append(scenario)
        seen.add(scenario)

    logger.info(f"R4 finalists assembled ({len(finalists)}): {finalists}")
    return finalists


def run_round_selection(df: pd.DataFrame) -> tuple[dict[str, list[str]], list[str]]:
    """
    Orchestrates per-round survivor selection for R0-R3 and assembles
    R4 finalists. All rounds are independent - no round output is used
    as input to another round's gating or scoring.

    Universal gating reference (computed once, used by R1, R2, R3):
        df_p_raw = df[df["scenario_canonical"] == GATING_REFERENCE]
        p_raw_recall_min = df_p_raw["Recall"].min()
        If P_Raw_0 is not present in df, return empty round outputs with a
        clear warning instead of raising.
    """
    round_survivors = {"R0": DOE_PLAN["R0"].copy(), "R1": [], "R2": [], "R3": []}

    df_p_raw = df[df["scenario_canonical"] == GATING_REFERENCE].copy()
    if df_p_raw.empty:
        available = sorted(df["scenario_canonical"].dropna().unique().tolist())
        logger.warning(
            "Round selection skipped: universal gating reference "
            f"'{GATING_REFERENCE}' is missing from the results."
        )
        logger.warning(f"Available scenarios in the input results: {available}")
        logger.warning(
            "Expected at least one row for the gating reference to compute "
            "R1-R3 survivors and assemble R4 finalists."
        )
        return round_survivors, []
    p_raw_recall_min = float(df_p_raw["Recall"].astype(float).min())

    df_r0 = df[df["scenario_canonical"].isin(DOE_PLAN["R0"])].copy()
    for scenario in DOE_PLAN["R0"]:
        df_s = df_r0[df_r0["scenario_canonical"] == scenario].copy()
        if df_s.empty:
            logger.warning(f"R0 reference scenario missing: {scenario}")
            continue

        stats = {}
        for metric in METRICS:
            values = df_s[metric].astype(float)
            stats[f"{metric}_mean"] = f"{float(values.mean()):.4f}"
            stats[f"{metric}_min"] = f"{float(values.min()):.4f}"
        logger.info(f"R0 reference {scenario}: {stats}")

    df_r1 = df[df["scenario_canonical"].isin(DOE_PLAN["R1"])].copy()
    r1_survivors, r1_scores, _ = select_round_survivors(
        df_r1,
        "R1",
        p_raw_recall_min,
        max_survivors=2,
        force_if_empty=False,
    )
    r1_scores_fmt = {sc: f"{v:.4f}" for sc, v in r1_scores.items()}
    logger.info(f"R1 survivors ({len(r1_survivors)}): {r1_survivors}")
    logger.info(f"R1 scores: {r1_scores_fmt}")
    round_survivors["R1"] = r1_survivors

    df_r2 = df[df["scenario_canonical"].isin(DOE_PLAN["R2"])].copy()
    r2_survivors, r2_scores, r2_forced = select_round_survivors(
        df_r2,
        "R2",
        p_raw_recall_min,
        max_survivors=2,
        force_if_empty=True,
    )
    r2_scores_fmt = {sc: f"{v:.4f}" for sc, v in r2_scores.items()}
    logger.info(f"R2 survivors ({len(r2_survivors)}): {r2_survivors}")
    logger.info(f"R2 scores: {r2_scores_fmt}")
    if r2_forced:
        logger.info("R2 forced: True")
    round_survivors["R2"] = r2_survivors

    df_r3 = df[df["scenario_canonical"].isin(DOE_PLAN["R3"])].copy()
    r3_survivors, r3_scores, _ = select_round_survivors(
        df_r3,
        "R3",
        p_raw_recall_min,
        max_survivors=3,
        force_if_empty=False,
    )
    r3_scores_fmt = {sc: f"{v:.4f}" for sc, v in r3_scores.items()}
    logger.info(f"R3 survivors ({len(r3_survivors)}): {r3_survivors}")
    logger.info(f"R3 scores: {r3_scores_fmt}")
    round_survivors["R3"] = r3_survivors

    r4_finalists = assemble_r4_finalists(round_survivors, r2_forced)

    return round_survivors, r4_finalists


def run_doe_pipeline() -> None:
    print("\n" + "=" * 90)
    print("REDUCED DOE PIPELINE (R0-R3 + POST-R3 ROBUST SELECTION)")
    print("=" * 90)

    df = load_master_results()
    validate_master(df)

    df = df.copy()
    df["scenario_canonical"] = df["Escenario"].map(normalize_scenario)
    df = df[df["scenario_canonical"].isin(SCENARIO_ORDER)].copy()
    df["Round"] = df["scenario_canonical"].map(SCENARIO_TO_ROUND)
    FIGS_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    global_dir = OUTPUT_ROOT / "global"
    global_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        raise ValueError("No hay escenarios válidos del DOE reducido en el CSV de resultados.")

    # 1) Completeness report
    completeness = check_round_completeness(df)
    completeness_path = global_dir / "doe_round_completeness_reduced.csv"
    completeness.to_csv(completeness_path, index=False)
    logger.info(f"Completeness exportado: {completeness_path}")

    # 2) Round summary
    summary = build_round_summary(df)
    summary_path = global_dir / "doe_round_summary_reduced.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Resumen por rondas exportado: {summary_path}")

    # 3) Delta matrix vs baseline
    delta = build_delta_vs_baseline(df, baseline="B_Raw_0")
    if not delta.empty:
        delta_path = global_dir / "doe_delta_vs_baseline_reduced.csv"
        delta.to_csv(delta_path, index=False)
        logger.info(f"Delta vs baseline exportado: {delta_path}")

    # 4) Round artifacts: csv + figures + deltas
    for round_name, scenarios_round in DOE_PLAN.items():
        df_round = df[df["scenario_canonical"].isin(scenarios_round)].copy()
        if df_round.empty:
            logger.warning(f"{round_name}: sin datos para generar artefactos.")
            continue

        round_output_dir = OUTPUT_ROOT / round_name
        round_output_dir.mkdir(parents=True, exist_ok=True)
        round_fig_dir = round_output_dir / "figs"
        round_fig_dir.mkdir(parents=True, exist_ok=True)

        round_csv = round_output_dir / f"{round_name}_reduced.csv"
        df_round.to_csv(round_csv, index=False)

        summary_round = build_summary_for_round(df_round)
        summary_csv = round_output_dir / f"{round_name}_resumen_reduced.csv"
        summary_round.to_csv(summary_csv, index=False)

        plot_dataset_bars(df_round, round_name, round_fig_dir)
        plot_abs_summary_heatmap(
            summary_round,
            round_name,
            round_fig_dir / f"{round_name}_heatmap_resumen_abs.png",
        )
        generate_deltas_for_round(df_round, round_name, round_fig_dir)

    # 4b) Per-round survivor selection + R4 finalist assembly
    round_survivors, r4_finalists = run_round_selection(df)

    # Export per-round survivors table
    survivors_rows = []
    for rnd, sc_list in round_survivors.items():
        for sc in sc_list:
            survivors_rows.append(
                {"Round": rnd, "survivor": sc, "is_r4_finalist": sc in r4_finalists}
            )
    survivors_path = global_dir / "doe_round_survivors.csv"
    pd.DataFrame(survivors_rows).to_csv(survivors_path, index=False)
    logger.info(f"Sobrevivientes por ronda exportados: {survivors_path}")

    # Export R4 finalists table
    r4_rows = [{"r4_finalist": sc, "source_round": SCENARIO_TO_ROUND.get(sc, "")} for sc in r4_finalists]
    r4_path = global_dir / "doe_r4_finalists.csv"
    pd.DataFrame(r4_rows).to_csv(r4_path, index=False)
    logger.info(f"R4 finalists exportados: {r4_path}")

    if r4_finalists:
        print(f"\nFinalistas para R4: {r4_finalists}")
    else:
        print("\nFinalistas para R4: none")
        print(
            f"R4 finalist assembly was skipped because the gating reference "
            f"'{GATING_REFERENCE}' is missing from the results."
        )

    # 5) Post-R3 robust selection
    ranking = apply_gating_and_scoring(df)
    if ranking.empty:
        print(
            "\nPost-R3 ranking was not generated because required baseline "
            f"data for '{GATING_REFERENCE}' is missing."
        )
        print(
            f"Check the input CSV and make sure scenario '{GATING_REFERENCE}' "
            "is present with valid metric rows."
        )
        print("\n" + "=" * 90)
        print("PIPELINE COMPLETED WITH MISSING REQUIREMENTS")
        print("=" * 90)
        return

    ranking_path = global_dir / "doe_post_r3_ranking_reduced.csv"
    ranking.to_csv(ranking_path, index=False)
    logger.info(f"Ranking post-R3 exportado: {ranking_path}")
    plot_post_r3_ranking(ranking, global_dir / "post_r3_ranking.png")
    plot_post_r3_gating_threshold(ranking, global_dir / "post_r3_gating_threshold.png")

    print("\nTop candidatos (passed_gating=True):")
    if not ranking.empty:
        top = ranking[ranking["passed_gating"]].head(5)
        if top.empty:
            print("  Ningún escenario superó gating.")
        else:
            print(top[["Rank", "scenario_canonical", "S_global", "Recall_min"]].to_string(index=False))
    else:
        print("  Ranking vacío.")

    print("\n" + "=" * 90)
    print("PIPELINE COMPLETADO")
    print("=" * 90)


if __name__ == "__main__":
    utils.run_with_sqlite_registration(
        script_name="17_doe_pipeline_reduced.py",
        func=run_doe_pipeline,
        db_path=config.DB_PATH,
        outputs={"output_root": OUTPUT_ROOT, "figs_root": FIGS_ROOT},
    )
