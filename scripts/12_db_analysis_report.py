"""
12 DB ANALYSIS REPORT (portable dataset)
-----------------------------------------------------------------------
Generate database-level analysis by:
1) summarizing images, annotations, and QA labels
2) inspecting duplicate patterns and data quality signals
3) exporting tabular reports for review
-----------------------------------------------------------------------
"""

import sqlite3

import pandas as pd

import config
import config_analysis

try:
    import imagehash
except ImportError:
    imagehash = None

NOISE_LABEL_STATES = {"missing_label", "bad_label", "poor_alignment"}
JUDGE_ERROR_LABELS = {"bad_label", "poor_alignment"}
TRACKED_LABEL_ISSUES = ("bad_label", "missing_label", "poor_alignment")


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
    if imagehash is None:
        raise ImportError(
            "imagehash no est\xC3\xA1 instalado y se requiere para deduplicaci\xC3\xB3n fallback."
        )

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
    removed = len(df) - len(merged)
    return merged, removed


def apply_dedup(df: pd.DataFrame, use_dedup: bool) -> tuple[pd.DataFrame, int]:
    if not use_dedup or df.empty:
        return df, 0

    if "is_dedup_keep" in df.columns:
        before = len(df)
        out = df[df["is_dedup_keep"] == 1]
        return out, before - len(out)

    return deduplicate_by_hamming(df, threshold=config.HAMMING_THRESHOLD)


def merge_label_issues(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    # Preferred source in current one-pass pipeline.
    if "label_issue" in df_raw.columns:
        df_raw["label_issue"] = df_raw["label_issue"].fillna("unknown")
        return df_raw, "analysis_images"

    if not config.DB_PATH.exists():
        df_raw["label_issue"] = "unknown"
        return df_raw, "no_master_db"

    try:
        conn_master = sqlite3.connect(config.DB_PATH)
        try:
            df_qa = pd.read_sql(
                "SELECT id AS qa_image_id, relative_path, label_issue FROM images", conn_master
            )
        finally:
            conn_master.close()

        df = df_raw.merge(df_qa, on="relative_path", how="left")
        df["label_issue"] = df["label_issue"].fillna("unknown")

        unknown_mask = df["label_issue"].eq("unknown")
        unknown_count = int(unknown_mask.sum())
        if unknown_count == len(df):
            audit_csv = config.EXPORTS_DIR / "audit_details.csv"
            if audit_csv.exists():
                df_audit = pd.read_csv(audit_csv)
                if {"image_id", "status"}.issubset(df_audit.columns):
                    df_audit = df_audit.rename(columns={"status": "label_issue_audit"})
                    df = df.merge(
                        df_audit[["image_id", "label_issue_audit"]],
                        left_on="qa_image_id",
                        right_on="image_id",
                        how="left",
                    )

                    fill_mask = df["label_issue"].eq("unknown") & df["label_issue_audit"].notna()
                    filled = int(fill_mask.sum())
                    df.loc[fill_mask, "label_issue"] = df.loc[fill_mask, "label_issue_audit"]
                    df = df.drop(columns=["image_id", "label_issue_audit"], errors="ignore")
                    remaining_unknown = int(df["label_issue"].eq("unknown").sum())

                    print(
                        f"[INFO] label_issue source: audit_details.csv fallback | "
                        f"filled={filled}, unknown_remaining={remaining_unknown}"
                    )
                    return df, "audit_csv_fallback"

        return df, "master_images"

    except Exception as e:
        print(f"[WARN] Could not merge QA labels: {e}")
        df_raw["label_issue"] = "unknown"
        return df_raw, "merge_error"


def count_label_issues(df_in: pd.DataFrame) -> dict[str, int]:
    return {
        issue: int((df_in["label_issue"] == issue).sum())
        for issue in TRACKED_LABEL_ISSUES
    }


def main():
    print("=" * 110)
    print("DB ANALYSIS REPORT (DOE-ALIGNED)")
    print("=" * 110)

    if not config_analysis.ANALYSIS_DB_PATH.exists():
        print("[ERROR] analysis_metadata.sqlite not found.")
        return

    conn = sqlite3.connect(config_analysis.ANALYSIS_DB_PATH)
    try:
        df_raw = pd.read_sql("SELECT * FROM analysis_images", conn)
    finally:
        conn.close()

    if df_raw.empty:
        print("[ERROR] analysis_images is empty.")
        return

    df_raw, label_source = merge_label_issues(df_raw)
    print(f"[INFO] label_issue source used: {label_source}")

    total_raw = len(df_raw)
    scenario_results = []

    for scenario_name, params in config_analysis.SCENARIOS.items():
        if not params.get("build_dataset", True):
            scenario_results.append(
                {
                    "Scenario": scenario_name,
                    "ID": params.get("id", ""),
                    "Domain": params.get("domain", ""),
                    "Judge tau": float(params.get("judge_thresh", 0.0)),
                    "Base Fuente": 0,
                    "SemNoise Out": 0,
                    "Sem MissingLabel": 0,
                    "Sem BadLabel": 0,
                    "Sem PoorAlign": 0,
                    "Post-pHash": 0,
                    "Judge Out": 0,
                    "Judge MissingLabel": 0,
                    "Judge BadLabel": 0,
                    "Judge PoorAlign": 0,
                    "Judge WeakValid": 0,
                    "Post-Judge": 0,
                    "Post-Clean": 0,
                    "Injected": 0,
                    "Final Total": 0,
                    "Retention %": 0.0,
                    "Note": "Metadata-only control",
                }
            )
            continue

        sources = resolve_sources(params)
        df_source = df_raw[df_raw["source"].isin(sources)].copy()
        base_source = len(df_source)

        df_post_phash, _ = apply_dedup(df_source, bool(params.get("dedup", False)))
        post_phash = len(df_post_phash)

        judge_bad_label = 0
        judge_missing_label = 0
        judge_poor_alignment = 0
        judge_weak_valid = 0
        judge_out = 0
        post_judge = post_phash

        thresh = float(params.get("judge_thresh", 0.0))
        if thresh > 0:
            score_ok = df_post_phash["judge_score"].fillna(-1e9) >= thresh
            removed = df_post_phash[~score_ok]

            judge_bad_label = int((removed["label_issue"] == "bad_label").sum())
            judge_missing_label = int((removed["label_issue"] == "missing_label").sum())
            judge_poor_alignment = int((removed["label_issue"] == "poor_alignment").sum())
            judge_weak_valid = int((~removed["label_issue"].isin(JUDGE_ERROR_LABELS | {"missing_label"})).sum())

            judge_out = len(removed)
            post_judge = len(df_post_phash[score_ok])

        if thresh > 0:
            sem_counts = count_label_issues(df_post_phash)
            sem_out = sum(sem_counts.values())
            df_final = df_post_phash[~df_post_phash["label_issue"].isin(NOISE_LABEL_STATES)]
            df_final = df_final[df_final["judge_score"].fillna(-1e9) >= thresh]
        else:
            sem_counts = {issue: 0 for issue in TRACKED_LABEL_ISSUES}
            sem_out = 0
            df_final = df_post_phash.copy()

        post_clean = len(df_final)
        train_base = int(post_clean * config.SPLIT_RATIOS[0])
        noise_rate = float(params.get("noise_rate", 0.0))
        # Must mirror 13_dataset_generation.py: noise is sampled only from TRAIN split.
        injected = int(train_base * noise_rate)
        final_total = post_clean + injected

        retention_pct = round((final_total / base_source) * 100, 1) if base_source else 0.0

        scenario_results.append(
            {
                "Scenario": scenario_name,
                "ID": params.get("id", ""),
                "Domain": params.get("domain", ""),
                "Judge tau": thresh,
                "Base Fuente": base_source,
                "SemNoise Out": sem_out,
                "Sem MissingLabel": sem_counts["missing_label"],
                "Sem BadLabel": sem_counts["bad_label"],
                "Sem PoorAlign": sem_counts["poor_alignment"],
                "Post-pHash": post_phash,
                "Judge Out": judge_out,
                "Judge MissingLabel": judge_missing_label,
                "Judge BadLabel": judge_bad_label,
                "Judge PoorAlign": judge_poor_alignment,
                "Judge WeakValid": judge_weak_valid,
                "Post-Judge": post_judge,
                "Post-Clean": post_clean,
                "Train Base": train_base,
                "Injected": injected,
                "Final Total": final_total,
                "Retention %": retention_pct,
                "Note": "Judge split computed before semantic exclusion",
            }
        )

    df_res = pd.DataFrame(scenario_results)

    cols = [
        "Scenario",
        "ID",
        "Domain",
        "Judge tau",
        "Base Fuente",
        "SemNoise Out",
        "Sem MissingLabel",
        "Sem BadLabel",
        "Sem PoorAlign",
        "Post-pHash",
        "Judge Out",
        "Judge MissingLabel",
        "Judge BadLabel",
        "Judge PoorAlign",
        "Judge WeakValid",
        "Post-Judge",
        "Post-Clean",
        "Train Base",
        "Injected",
        "Final Total",
        "Retention %",
        "Note",
    ]

    print("\n" + "=" * 130)
    print("BALANCE BY SCENARIO (SOURCE -> pHASH -> JUDGE -> SEMANTIC EXCLUSION -> INJECTION)")
    print("=" * 130)
    try:
        from tabulate import tabulate

        print(tabulate(df_res[cols], headers="keys", tablefmt="github", showindex=False))
    except ImportError:
        print(df_res[cols].to_string(index=False))

    out_csv = config.EXPORTS_DIR / "db_analysis_report.csv"
    df_res.to_csv(out_csv, index=False)

    tau_focus = sorted({config_analysis.JUDGE_RELAX, config_analysis.JUDGE_STRICT})
    df_tau = (
        df_res[df_res["Judge tau"].isin(tau_focus)]
        .copy()
        .sort_values(["Judge tau", "Domain", "Scenario"])
    )

    tau_cols = [
        "Scenario",
        "ID",
        "Domain",
        "Judge tau",
        "Base Fuente",
        "Post-pHash",
        "Judge Out",
        "Judge MissingLabel",
        "Judge BadLabel",
        "Judge PoorAlign",
        "Judge WeakValid",
        "Sem MissingLabel",
        "Sem BadLabel",
        "Sem PoorAlign",
        "Post-Judge",
        "Post-Clean",
        "Final Total",
    ]

    print("\n" + "=" * 130)
    print("THRESHOLD BREAKDOWN (tau = 0.35 / 0.65)")
    print("=" * 130)
    try:
        from tabulate import tabulate

        print(tabulate(df_tau[tau_cols], headers="keys", tablefmt="github", showindex=False))
    except ImportError:
        print(df_tau[tau_cols].to_string(index=False))

    out_tau_csv = config.EXPORTS_DIR / "db_threshold_breakdown.csv"
    df_tau[tau_cols].to_csv(out_tau_csv, index=False)
    print("-" * 130)
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_tau_csv}")


if __name__ == "__main__":
    main()
