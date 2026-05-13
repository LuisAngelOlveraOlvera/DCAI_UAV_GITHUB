"""
11 SCENARIO SIMULATOR (DoE pipeline)
-----------------------------------------------------------------------
Simulate structured DoE scenarios from metadata by:
1) applying source filters and judge thresholds
2) excluding semantic noise and deduplicating by pHash/Hamming
3) sweeping scenario settings without creating physical datasets
4) exporting comparative CSV reports
-----------------------------------------------------------------------
"""

import argparse
import sqlite3

import imagehash
import pandas as pd

import config
import config_analysis
import utils

logger = utils.setup_logger(
    "Scenario_Sim",
    log_file=str(config.LOGS_DIR / "scenario_simulation.log"),
)

NOISE_LABEL_STATES = {"missing_label", "bad_label", "poor_alignment"}
SIMULATION_HAMMING_THRESHOLDS = (1, 3, 5, 10)


def resolve_sources(params):
    if "sources" in params:
        return list(params.get("sources") or [])
    sources = []
    if params.get("use_pascal"):
        sources.append("PASCAL")
    if params.get("use_propio"):
        sources.append("PROPIO")
    return sources


def deduplicate_by_hamming(df: pd.DataFrame, threshold: int) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0

    work = df.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    valid_mask = work["phash"].notna() & (work["phash"].astype(str) != "")
    valid_df = work[valid_mask].copy()
    invalid_df = work[~valid_mask].copy()

    if valid_df.empty:
        return work.drop(columns=["_orig_idx"]).reset_index(drop=True), 0

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
        ra, rb = find(a), find(b)
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

    removed = len(df) - len(merged)
    return merged, removed


def get_risk_assessment(row):
    risks = []
    if row["Ruido Agregado"] > 0:
        risks.append("Leakage/Overfitting (Alto)")
    if row["Total Final"] < 1000:
        risks.append("Data Starvation (Critico)")
    elif row["Total Final"] < 3000:
        risks.append("Data Starvation (Moderado)")
    if row["Judge tau"] < 0.2:
        risks.append("Ruido de Etiquetas (Posible)")
    return " + ".join(risks) if risks else "Bajo / Balanceado"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simula escenarios DOE con barrido de umbrales Hamming pHash."
    )
    parser.add_argument(
        "--hamming-thresholds",
        nargs="+",
        type=int,
        default=list(SIMULATION_HAMMING_THRESHOLDS),
        help="Lista de umbrales Hamming a simular. Ejemplo: --hamming-thresholds 1 3 5 10",
    )
    return parser.parse_args()


def run_simulation(hamming_thresholds=None):
    thresholds = sorted({int(t) for t in (hamming_thresholds or SIMULATION_HAMMING_THRESHOLDS)})
    logger.info("=== START DOE SCENARIO SIMULATION ===")
    logger.info(f"Hamming thresholds to simulate: {thresholds}")

    if not config_analysis.ANALYSIS_DB_PATH.exists():
        logger.error("Analysis DB not found. Run 02_analysis_etl.py and 07_judge_scoring.py first.")
        return

    conn = sqlite3.connect(config_analysis.ANALYSIS_DB_PATH)
    try:
        df_raw = pd.read_sql("SELECT * FROM analysis_images", conn)
    finally:
        conn.close()

    if df_raw.empty:
        logger.error("analysis_images is empty.")
        return

    # Prefer label_issue already computed in analysis DB (one-pass Judge).
    if "label_issue" in df_raw.columns:
        df_raw["label_issue"] = df_raw["label_issue"].fillna("unknown")
        logger.info("QA source: analysis_images.label_issue")
    elif config.DB_PATH.exists():
        try:
            conn_master = sqlite3.connect(config.DB_PATH)
            try:
                df_qa = pd.read_sql("SELECT relative_path, label_issue FROM images", conn_master)
            finally:
                conn_master.close()
            df_raw = df_raw.merge(df_qa, on="relative_path", how="left")
            df_raw["label_issue"] = df_raw["label_issue"].fillna("unknown")
            logger.info("QA source: dataset_master.images.label_issue")
        except Exception as e:
            logger.warning(f"QA merge skipped: {e}")
            df_raw["label_issue"] = "unknown"
    else:
        df_raw["label_issue"] = "unknown"

    qa_counts = df_raw["label_issue"].value_counts(dropna=False).to_dict()
    logger.info(f"QA distribution: {qa_counts}")

    total_raw = len(df_raw)
    results = []
    source_cache = {}
    dedup_cache = {}

    def get_source_slice(sources):
        source_key = tuple(sorted(sources))
        if source_key not in source_cache:
            df_source = df_raw[df_raw["source"].isin(source_key)].copy()
            count_after_source = len(df_source)
            noise_removed = int(df_source["label_issue"].isin(NOISE_LABEL_STATES).sum())
            source_cache[source_key] = (df_source, count_after_source, noise_removed)
        return source_cache[source_key]

    for hamming_threshold in thresholds:
        for scenario_name, params in config_analysis.SCENARIOS.items():
            if not params.get("build_dataset", True):
                results.append(
                    {
                        "Hamming tau": hamming_threshold,
                        "Scenario": scenario_name,
                        "ID": params.get("id", ""),
                        "Domain": params.get("domain", ""),
                        "Dedup pHash": params.get("dedup", False),
                        "Judge tau": float(params.get("judge_thresh", 0.0)),
                        "Noise %": int(float(params.get("noise_rate", 0.0)) * 100),
                        "Base Fuente": 0,
                        "Noise Semantico Excluido": 0,
                        "Deduplicadas": 0,
                        "Rechazo Juez": 0,
                        "Clean": 0,
                        "Ruido Agregado": 0,
                        "Total Final": 0,
                        "Retencion %": 0.0,
                        "Diagnostico Riesgo": "Control metadata-only",
                    }
                )
                continue

            sources = resolve_sources(params)
            df_curr, count_after_source, noise_removed = get_source_slice(sources)
            df_curr = df_curr.copy()

            dups_removed = 0
            if params.get("dedup", False):
                dedup_key = (tuple(sorted(sources)), hamming_threshold)
                if dedup_key not in dedup_cache:
                    dedup_cache[dedup_key] = deduplicate_by_hamming(
                        df_curr, threshold=hamming_threshold
                    )
                df_curr, dups_removed = dedup_cache[dedup_key]
                df_curr = df_curr.copy()

            judge_removed = 0
            thresh = float(params.get("judge_thresh", 0.0))
            if thresh > 0:
                df_curr = df_curr[~df_curr["label_issue"].isin(NOISE_LABEL_STATES)].copy()
            else:
                noise_removed = 0

            if thresh > 0:
                before = len(df_curr)
                df_curr = df_curr[df_curr["judge_score"] >= thresh]
                judge_removed = before - len(df_curr)

            clean_count = len(df_curr)
            noise_rate = float(params.get("noise_rate", 0.0))
            noise_added = int(clean_count * noise_rate)
            total_final = clean_count + noise_added

            row = {
                "Hamming tau": hamming_threshold,
                "Scenario": scenario_name,
                "ID": params.get("id", ""),
                "Domain": params.get("domain", ""),
                "Dedup pHash": params.get("dedup", False),
                "Judge tau": thresh,
                "Noise %": int(noise_rate * 100),
                "Base Fuente": count_after_source,
                "Noise Semantico Excluido": noise_removed,
                "Deduplicadas": dups_removed,
                "Rechazo Juez": judge_removed,
                "Clean": clean_count,
                "Ruido Agregado": noise_added,
                "Total Final": total_final,
                "Retencion %": round((total_final / total_raw) * 100, 1) if total_raw else 0.0,
            }
            row["Diagnostico Riesgo"] = get_risk_assessment(row)
            results.append(row)

    df_results = pd.DataFrame(results)
    df_results.to_csv(config_analysis.REPORT_CSV, index=False)

    print("\n" + "=" * 120)
    print("TABLA COMPARATIVA DOE (SIMULACION)")
    print("=" * 120)
    try:
        from tabulate import tabulate

        print(tabulate(df_results, headers="keys", tablefmt="github", showindex=False))
    except ImportError:
        print(df_results.to_string(index=False))
    print("=" * 120)
    print(f"Reporte guardado en: {config_analysis.REPORT_CSV}")


if __name__ == "__main__":
    args = parse_args()
    run_simulation(hamming_thresholds=args.hamming_thresholds)
