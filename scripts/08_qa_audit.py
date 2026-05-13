"""
08 QA AUDIT SYNC (DoE pipeline)
-----------------------------------------------------------------------
Reuse judge results without inference by:
1) copying label_issue values into dataset_master.sqlite
2) synchronizing QA audit metadata
3) exporting audit_details.csv for traceability
-----------------------------------------------------------------------
"""

import sqlite3
import sys

import pandas as pd

import config
import config_analysis
import utils

logger = utils.setup_logger("QA_Audit_Sync", log_file=str(config.LOGS_DIR / "qa_audit.log"))


def has_column(conn, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


def run_audit_sync():
    logger.info("--> INICIANDO QA SYNC (SIN REINFERENCIA) <--")
    max_unknown_pct = float(getattr(config, "QA_MAX_UNKNOWN_PCT", 5.0))

    if not config_analysis.ANALYSIS_DB_PATH.exists():
        logger.error(f"No existe DB de análisis: {config_analysis.ANALYSIS_DB_PATH}")
        logger.error("Ejecuta primero: python 07_judge_scoring.py")
        return

    if not config.DB_PATH.exists():
        logger.error(f"No existe DB maestra: {config.DB_PATH}")
        logger.error("Ejecuta primero: python 04_etl_ingestion.py")
        return

    conn_analysis = sqlite3.connect(config_analysis.ANALYSIS_DB_PATH)
    try:
        if not has_column(conn_analysis, "analysis_images", "label_issue"):
            logger.error("analysis_images no tiene columna label_issue.")
            logger.error("Ejecuta nuevamente: python 07_judge_scoring.py")
            return

        df_qa = pd.read_sql(
            """
            SELECT relative_path, label_issue
            FROM analysis_images
            WHERE label_issue IS NOT NULL
            """,
            conn_analysis,
        )
    finally:
        conn_analysis.close()

    if df_qa.empty:
        logger.error("No hay resultados QA en analysis_images.")
        logger.error("Ejecuta: python 07_judge_scoring.py")
        return

    conn_master = sqlite3.connect(config.DB_PATH)
    cur_master = conn_master.cursor()

    try:
        updates = [(row.label_issue, row.relative_path) for row in df_qa.itertuples(index=False)]
        cur_master.executemany(
            "UPDATE images SET label_issue = ? WHERE relative_path = ?",
            updates,
        )
        conn_master.commit()

        df_stats = pd.read_sql(
            "SELECT label_issue, COUNT(*) AS n FROM images GROUP BY label_issue ORDER BY n DESC",
            conn_master,
        )
        total_images = int(df_stats["n"].sum()) if not df_stats.empty else 0
        unknown_count = int(
            df_stats.loc[df_stats["label_issue"] == "unknown", "n"].sum()
        ) if not df_stats.empty else 0
        unknown_pct = (unknown_count / total_images * 100.0) if total_images else 0.0

        df_audit = pd.read_sql(
            "SELECT id AS image_id, label_issue AS status FROM images",
            conn_master,
        )
        csv_path = config.EXPORTS_DIR / "audit_details.csv"
        df_audit.to_csv(csv_path, index=False)

        logger.info("=" * 55)
        logger.info("RESULTADOS QA SYNC")
        logger.info("=" * 55)
        for row in df_stats.itertuples(index=False):
            logger.info(f"{row.label_issue}: {row.n}")
        logger.info(
            f"unknown={unknown_count}/{total_images} ({unknown_pct:.2f}%) | "
            f"límite={max_unknown_pct:.2f}%"
        )
        if unknown_pct > max_unknown_pct:
            logger.critical(
                "QA inválido: porcentaje de 'unknown' supera el límite permitido. "
                "Revisa 07_judge_scoring.py y vuelve a ejecutar."
            )
            raise RuntimeError(
                f"unknown_pct={unknown_pct:.2f}% > QA_MAX_UNKNOWN_PCT={max_unknown_pct:.2f}%"
            )
        logger.info("=" * 55)
        logger.info(f"[REPORTE] Detalles guardados en: {csv_path}")

    except Exception as e:
        logger.critical(f"Fallo en QA sync: {e}", exc_info=True)
    finally:
        conn_master.close()
        logger.info("Conexiones cerradas.")


if __name__ == "__main__":
    try:
        run_audit_sync()
    except Exception:
        sys.exit(1)
