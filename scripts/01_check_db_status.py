"""
01 CHECK DB STATUS (DoE pipeline)
-----------------------------------------------------------------------
Quick health check for:
1) analysis_metadata.sqlite (pHash, dedup, judge)
2) dataset_master.sqlite (images, annotations, QA labels)
-----------------------------------------------------------------------
"""

import sqlite3
from pathlib import Path

import config
import config_analysis


def table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def table_columns(conn, name: str):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({name})").fetchall()]


def print_analysis_status():
    db = config_analysis.ANALYSIS_DB_PATH
    print("=" * 90)
    print("ANALYSIS DB STATUS")
    print("=" * 90)
    print(f"Path: {db}")
    if not db.exists():
        print("[MISSING] analysis_metadata.sqlite")
        print("Next: python scripts/02_analysis_etl.py")
        return

    conn = sqlite3.connect(db)
    try:
        if not table_exists(conn, "analysis_images"):
            print("[ERROR] Table analysis_images not found.")
            print("Next: python scripts/02_analysis_etl.py")
            return

        cols = table_columns(conn, "analysis_images")
        total = conn.execute("SELECT COUNT(*) FROM analysis_images").fetchone()[0]
        print(f"Rows analysis_images: {total}")

        if "phash" in cols:
            phash_not_null = conn.execute(
                "SELECT COUNT(*) FROM analysis_images WHERE phash IS NOT NULL AND phash != ''"
            ).fetchone()[0]
            print(f"pHash populated: {phash_not_null}/{total}")

        if "is_dedup_keep" in cols:
            keep = conn.execute("SELECT COUNT(*) FROM analysis_images WHERE is_dedup_keep=1").fetchone()[0]
            drop = conn.execute("SELECT COUNT(*) FROM analysis_images WHERE is_dedup_keep=0").fetchone()[0]
            print(f"Dedup keep/drop: {keep}/{drop}")
        else:
            print("Dedup flags: [MISSING] is_dedup_keep")

        if "judge_score" in cols:
            scored = conn.execute("SELECT COUNT(*) FROM analysis_images WHERE judge_score > 0").fetchone()[0]
            print(f"Judge scored (>0): {scored}/{total}")
        else:
            print("Judge score column: [MISSING] judge_score")

        if "processed" in cols:
            processed = conn.execute("SELECT COUNT(*) FROM analysis_images WHERE processed != 0").fetchone()[0]
            pending = conn.execute("SELECT COUNT(*) FROM analysis_images WHERE processed = 0").fetchone()[0]
            print(f"Judge processed/pending: {processed}/{pending}")
    finally:
        conn.close()


def print_master_status():
    db = config.DB_PATH
    print("=" * 90)
    print("MASTER DB STATUS")
    print("=" * 90)
    print(f"Path: {db}")
    if not db.exists():
        print("[MISSING] dataset_master.sqlite")
        print("Next: python scripts/04_etl_ingestion.py")
        return

    conn = sqlite3.connect(db)
    try:
        has_images = table_exists(conn, "images")
        has_ann = table_exists(conn, "annotations")

        if not has_images:
            print("[ERROR] Table images not found.")
            print("Next: python scripts/04_etl_ingestion.py")
            return
        if not has_ann:
            print("[ERROR] Table annotations not found.")
            print("Next: python scripts/04_etl_ingestion.py")
            return

        total_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        total_ann = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
        print(f"Rows images: {total_images}")
        print(f"Rows annotations: {total_ann}")

        cols = table_columns(conn, "images")
        if "label_issue" in cols:
            print("label_issue distribution:")
            rows = conn.execute(
                "SELECT label_issue, COUNT(*) FROM images GROUP BY label_issue ORDER BY COUNT(*) DESC"
            ).fetchall()
            for status, cnt in rows:
                print(f"  - {status}: {cnt}")
        else:
            print("QA status column: [MISSING] label_issue")
    finally:
        conn.close()


def print_pipeline_hint():
    print("=" * 90)
    print("PIPELINE HINT")
    print("=" * 90)
    print("Order:")
    print("1) python scripts/01_check_db_status.py")
    print("2) python scripts/02_analysis_etl.py")
    print("3) python scripts/03_hamming_source_report.py   # opcional")
    print("4) python scripts/04_etl_ingestion.py")
    print("5) python scripts/05_validar_integridad.py")
    print("6) python scripts/06_exportar_anotacion.py      # opcional")
    print("7) python scripts/07_judge_scoring.py")
    print("8) python scripts/08_qa_audit.py")
    print("9) python scripts/11_scenario_simulator.py")
    print("10) python scripts/12_db_analysis_report.py     # opcional")
    print("11) python scripts/13_dataset_generation.py")
    print("12) python scripts/14_validation_report.py")
    print("13) python scripts/15_training.py")
    print("14) python scripts/16_evaluar_coco_persona.py")


def main():
    print_analysis_status()
    print_master_status()
    print_pipeline_hint()


if __name__ == "__main__":
    main()

