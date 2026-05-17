"""
13 DATASET GENERATION (DoE pipeline)
-----------------------------------------------------------------------
Materialize physical DoE datasets by:
1) reading scenario definitions from config_analysis
2) applying source filters, judge thresholds, and Hamming deduplication
3) building Train/Val/Test splits
4) injecting synthetic noise only into Train when configured
-----------------------------------------------------------------------
"""

import argparse
import sqlite3
import shutil
import time

import imagehash
import pandas as pd
from tqdm import tqdm

import config
import config_analysis
import utils

# ============================================================
# LOGGER
# ============================================================
logger = utils.setup_logger(
    "Dataset_Construct",
    log_file=str(config.LOGS_DIR / "dataset_construction.log"),
)

NOISE_LABEL_STATES = {"missing_label", "bad_label", "poor_alignment"}


def generate_yolo_yaml(dataset_path, nc=1, names=["person"]):
    """Create the data.yaml file required by YOLO."""
    yaml_content = (
        f"path: {dataset_path.absolute().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {nc}\n"
        f"names: {names}\n"
    )
    with open(dataset_path / "data.yaml", "w") as f:
        f.write(yaml_content)


def copy_file_pair(src_img, dst_img_dir, dst_lbl_dir, new_name=None):
    """Copy image and label. If new_name is provided, rename destination files."""
    try:
        fname = src_img.name
        stem = src_img.stem
        suffix = src_img.suffix

        if new_name:
            dst_img_name = new_name + suffix
            dst_lbl_name = new_name + ".txt"
        else:
            dst_img_name = fname
            dst_lbl_name = stem + ".txt"

        src_lbl = config.LABELS_DIR / f"{stem}.txt"
        if not src_lbl.exists():
            logger.warning(f"Skipped image without label: {src_img}")
            return False

        shutil.copy2(src_img, dst_img_dir / dst_img_name)
        shutil.copy2(src_lbl, dst_lbl_dir / dst_lbl_name)

        return True
    except Exception as e:
        logger.error(f"Error copying {src_img.name}: {e}")
        return False


def resolve_sources(params):
    """Support new and legacy scenario formats."""
    if "sources" in params:
        return list(params.get("sources") or [])

    sources = []
    if params.get("use_pascal"):
        sources.append("PASCAL")
    if params.get("use_propio"):
        sources.append("PROPIO")
    return sources


def deduplicate_by_hamming(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """
    Deduplicate by pHash similarity:
    two samples are considered duplicates if Hamming distance <= threshold.
    """
    if df.empty:
        return df

    work = df.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    valid_mask = work["phash"].notna() & (work["phash"].astype(str) != "")
    valid_df = work[valid_mask].copy()
    invalid_df = work[~valid_mask].copy()

    if valid_df.empty:
        return work.drop(columns=["_orig_idx"]).reset_index(drop=True)

    hash_objs = []
    for h in valid_df["phash"]:
        try:
            hash_objs.append(imagehash.hex_to_hash(str(h)))
        except Exception:
            hash_objs.append(None)

    n = len(valid_df)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        h_i = hash_objs[i]
        if h_i is None:
            continue
        for j in range(i + 1, n):
            h_j = hash_objs[j]
            if h_j is None:
                continue
            if (h_i - h_j) <= threshold:
                union(i, j)

    representative = {}
    for i in range(n):
        root = find(i)
        if root not in representative:
            representative[root] = i

    keep_positions = sorted(representative.values())
    dedup_valid = valid_df.iloc[keep_positions]

    merged = pd.concat([dedup_valid, invalid_df], axis=0)
    merged = merged.sort_values("_orig_idx").drop(columns=["_orig_idx"]).reset_index(drop=True)
    return merged


def exclude_semantic_noise(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove rows flagged as semantic noise in dataset_master.images.label_issue.
    If QA info is unavailable, keeps the dataframe unchanged.
    """
    if df.empty or not config.DB_PATH.exists():
        return df, 0

    try:
        conn = sqlite3.connect(config.DB_PATH)
        try:
            df_qa = pd.read_sql_query("SELECT relative_path, label_issue FROM images", conn)
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"QA merge skipped (cannot read label_issue): {e}")
        return df, 0

    merged = df.merge(df_qa, on="relative_path", how="left")
    noise_mask = merged["label_issue"].isin(NOISE_LABEL_STATES)
    removed = int(noise_mask.sum())
    keep_cols = ["relative_path", "phash"]
    if "is_dedup_keep" in merged.columns:
        keep_cols.append("is_dedup_keep")
    filtered = merged.loc[~noise_mask, keep_cols].reset_index(drop=True)
    return filtered, removed


def build_scenario(scenario_name, params):
    t0 = time.perf_counter()
    if not params.get("build_dataset", True):
        logger.info(f"Skipping {scenario_name}: metadata-only scenario.")
        return {
            "scenario": scenario_name,
            "status": "skipped_metadata_only",
            "total_images": 0,
            "duration_sec": 0.0,
            "kpi_under_30s": True,
        }

    scenario_id = params.get("id", scenario_name)
    model_tag = params.get("model", scenario_name)
    out_dir = config.EXPORTS_DIR / scenario_name

    logger.info(f"\n{'='*70}")
    logger.info(f"BUILDING SCENARIO: {scenario_name}")
    logger.info(f"ID={scenario_id} | MODEL={model_tag} | DESC={params.get('description', '')}")
    logger.info(f"{'='*70}")

    utils.ensure_directory(out_dir, clean=True)

    for split in ["train", "val", "test"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    sources = resolve_sources(params)
    if not sources:
        logger.error(f"Scenario {scenario_name} has no valid sources configured.")
        duration = time.perf_counter() - t0
        return {
            "scenario": scenario_name,
            "status": "failed_no_sources",
            "total_images": 0,
            "duration_sec": round(duration, 4),
            "kpi_under_30s": duration < 30.0,
        }

    conn = sqlite3.connect(config_analysis.ANALYSIS_DB_PATH)
    try:
        cols_info = conn.execute("PRAGMA table_info(analysis_images)").fetchall()
        has_dedup_keep = any(c[1] == "is_dedup_keep" for c in cols_info)

        placeholders = ",".join(["?"] * len(sources))
        select_cols = "relative_path, phash"
        if has_dedup_keep:
            select_cols += ", is_dedup_keep"
        query = f"SELECT {select_cols} FROM analysis_images WHERE source IN ({placeholders})"
        query_params = list(sources)

        judge_thresh = float(params.get("judge_thresh", 0.0))
        if judge_thresh > 0:
            query += " AND judge_score >= ?"
            query_params.append(judge_thresh)

        df = pd.read_sql_query(query, conn, params=query_params)
    finally:
        conn.close()

    if judge_thresh > 0:
        df, removed_noise = exclude_semantic_noise(df)
        if removed_noise > 0:
            logger.info(f"Semantic noise removed (missing/bad/poor): {removed_noise}")
    else:
        removed_noise = 0

    if params.get("dedup", False):
        before = len(df)
        if "is_dedup_keep" in df.columns:
            df = df[df["is_dedup_keep"] == 1].copy()
            logger.info(f"Dedup (precomputed Hamming<={config.HAMMING_THRESHOLD}): {before} -> {len(df)}")
        else:
            threshold = int(params.get("hamming_threshold", config.HAMMING_THRESHOLD))
            df = deduplicate_by_hamming(df, threshold=threshold)
            logger.info(f"Dedup fallback (Hamming<={threshold}): {before} -> {len(df)}")

    if df.empty:
        logger.error(f"Scenario {scenario_name} produced an empty dataset.")
        duration = time.perf_counter() - t0
        return {
            "scenario": scenario_name,
            "status": "failed_empty_dataset",
            "total_images": 0,
            "duration_sec": round(duration, 4),
            "kpi_under_30s": duration < 30.0,
        }

    df = df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)

    n = len(df)
    train_end = int(n * config.SPLIT_RATIOS[0])
    val_end = int(n * (config.SPLIT_RATIOS[0] + config.SPLIT_RATIOS[1]))
    splits = {
        "train": df.iloc[:train_end],
        "val": df.iloc[train_end:val_end],
        "test": df.iloc[val_end:],
    }

    logger.info(
        f"Split: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}"
    )

    for split_name, split_df in splits.items():
        dst_img_dir = out_dir / "images" / split_name
        dst_lbl_dir = out_dir / "labels" / split_name
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Copy {split_name}"):
            src_path = config.DATASET_ROOT / row["relative_path"]
            if src_path.exists():
                copy_file_pair(src_path, dst_img_dir, dst_lbl_dir)

    noise_rate = float(params.get("noise_rate", 0.0))
    if noise_rate > 0:
        logger.info(f"Injecting synthetic noise in TRAIN: {noise_rate*100:.0f}%")
        n_train = len(splits["train"])
        n_noise = int(n_train * noise_rate)

        if n_train > 0 and n_noise > 0:
            noise_df = splits["train"].sample(n=n_noise, replace=True, random_state=config.SEED)
            dst_img_dir = out_dir / "images" / "train"
            dst_lbl_dir = out_dir / "labels" / "train"

            count = 0
            for i, (_, row) in tqdm(
                enumerate(noise_df.iterrows()), total=n_noise, desc="Create synthetic duplicates"
            ):
                src_path = config.DATASET_ROOT / row["relative_path"]
                if src_path.exists():
                    new_name = f"{src_path.stem}_dup_{i:04d}"
                    copy_file_pair(src_path, dst_img_dir, dst_lbl_dir, new_name=new_name)
                    count += 1
            logger.info(f"Noise images added to TRAIN: {count}")

    generate_yolo_yaml(out_dir)
    logger.info(f"Scenario {scenario_name} finished at: {out_dir}")
    duration = time.perf_counter() - t0
    total_images = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    return {
        "scenario": scenario_name,
        "status": "ok",
        "total_images": int(total_images),
        "duration_sec": round(duration, 4),
        "kpi_under_30s": duration < 30.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Materialize physical DoE datasets for all scenarios or a selected subset."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="Scenario names to build, for example: N2_B_Raw_0 N3_H_Raw_0",
    )
    return parser.parse_args()


def resolve_selected_scenarios(requested_names):
    scenarios = config_analysis.SCENARIOS
    if not requested_names:
        return scenarios

    alias_to_name = {}
    for scenario_name, params in scenarios.items():
        alias_to_name[scenario_name] = scenario_name
        short_id = str(params.get("id", "")).strip()
        if short_id:
            alias_to_name.setdefault(short_id, scenario_name)

    selected_names = []
    unknown = []
    for raw_name in requested_names:
        resolved = alias_to_name.get(raw_name)
        if resolved is None:
            unknown.append(raw_name)
            continue
        if resolved not in selected_names:
            selected_names.append(resolved)

    if unknown:
        available_full = ", ".join(sorted(scenarios.keys()))
        available_short = ", ".join(
            sorted(str(params.get("id", "")).strip() for params in scenarios.values() if params.get("id"))
        )
        raise ValueError(
            f"Unknown scenario(s): {', '.join(unknown)}. "
            f"Available short ids: {available_short}. "
            f"Available full names: {available_full}"
        )

    return {name: scenarios[name] for name in selected_names}


def run_construction(selected_scenarios=None):
    logger.info("STARTING DOE DATASET CONSTRUCTION")
    logger.info(f"HAMMING_THRESHOLD={config.HAMMING_THRESHOLD}")
    if selected_scenarios:
        logger.info(f"Selected scenarios: {', '.join(selected_scenarios.keys())}")
    else:
        logger.info("Selected scenarios: ALL")

    built = 0
    skipped = 0
    timing_rows = []
    scenarios = selected_scenarios or config_analysis.SCENARIOS
    for scenario_name, params in scenarios.items():
        try:
            if not params.get("build_dataset", True):
                skipped += 1
            result = build_scenario(scenario_name, params)
            if result:
                timing_rows.append(result)
            if params.get("build_dataset", True):
                built += 1
        except Exception as e:
            logger.critical(f"Error while building {scenario_name}: {e}", exc_info=True)
            timing_rows.append(
                {
                    "scenario": scenario_name,
                    "status": "exception",
                    "total_images": 0,
                    "duration_sec": None,
                    "kpi_under_30s": False,
                }
            )

    if timing_rows:
        df_timing = pd.DataFrame(timing_rows)
        df_timing["kpi_target_sec"] = 30.0
        df_timing["sec_per_1k_images"] = df_timing.apply(
            lambda r: round((r["duration_sec"] / (r["total_images"] / 1000.0)), 4)
            if pd.notna(r["duration_sec"]) and r["total_images"] > 0
            else None,
            axis=1,
        )
        timing_csv = config.EXPORTS_DIR / "sqlite_timing_report.csv"
        df_timing.to_csv(timing_csv, index=False)
        ok_kpi = int((df_timing["kpi_under_30s"] == True).sum())
        logger.info(
            f"SQLite timing report saved: {timing_csv} | KPI<30s passed: {ok_kpi}/{len(df_timing)}"
        )

    logger.info("\n" + "=" * 70)
    logger.info(f"DONE. Built={built}, Skipped metadata-only={skipped}")
    logger.info(f"Output root: {config.EXPORTS_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    selected_scenarios = resolve_selected_scenarios(args.scenarios)
    run_construction(selected_scenarios=selected_scenarios)
