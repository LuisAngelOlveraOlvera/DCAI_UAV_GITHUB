"""
03 HAMMING SOURCE REPORT (DoE pipeline)
-----------------------------------------------------------------------
Explain Hamming-based duplicates by:
1) counting removable duplicates by source
2) summarizing cluster composition
3) reporting duplicate totals by cluster type
Requires: python 02_analysis_etl.py
-----------------------------------------------------------------------
"""

import sqlite3
import pandas as pd

import config
import config_analysis


def _has_columns(conn, table_name, required_cols):
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]
    return all(c in cols for c in required_cols)


def main():
    db_path = config_analysis.ANALYSIS_DB_PATH
    if not db_path.exists():
        print(f"[ERROR] No existe la BD de análisis: {db_path}")
        print("        Ejecuta primero: python 02_analysis_etl.py")
        return

    conn = sqlite3.connect(db_path)
    try:
        if not _has_columns(conn, "analysis_images", ["source", "dedup_cluster", "is_dedup_keep"]):
            print("[ERROR] Faltan columnas de deduplicación precomputada en analysis_images.")
            print("        Vuelve a ejecutar: python 02_analysis_etl.py")
            return

        df = pd.read_sql_query(
            """
            SELECT id, source, dedup_cluster, is_dedup_keep
            FROM analysis_images
            WHERE dedup_cluster IS NOT NULL
            """,
            conn,
        )
    finally:
        conn.close()

    if df.empty:
        print("[WARN] No hay datos para reportar.")
        return

    total = len(df)
    kept = int((df["is_dedup_keep"] == 1).sum())
    removed = total - kept

    # Duplicates removed by source
    removed_by_source = (
        df[df["is_dedup_keep"] == 0]["source"].value_counts().rename_axis("source").reset_index(name="removed")
    )

    # Cluster composition
    cluster_src = (
        df.groupby("dedup_cluster")["source"]
        .agg(lambda s: ",".join(sorted(set(s.tolist()))))
        .reset_index(name="cluster_sources")
    )
    cluster_src["cluster_type"] = cluster_src["cluster_sources"].map(
        lambda x: "MIXTO(PASCAL+PROPIO)" if "," in x else f"SOLO({x})"
    )
    cluster_type_counts = cluster_src["cluster_type"].value_counts()

    # Duplicates removed by cluster type
    removed_with_type = df.merge(cluster_src[["dedup_cluster", "cluster_type"]], on="dedup_cluster", how="left")
    removed_by_cluster_type = (
        removed_with_type[removed_with_type["is_dedup_keep"] == 0]["cluster_type"]
        .value_counts()
        .rename_axis("cluster_type")
        .reset_index(name="removed")
    )

    print("=" * 80)
    print("REPORTE 02.5 - ORIGEN DE DUPLICADOS (HAMMING)")
    print("=" * 80)
    print(f"Total imágenes evaluadas: {total}")
    print(f"Representantes conservados (is_dedup_keep=1): {kept}")
    print(f"Duplicados eliminables (is_dedup_keep=0): {removed}")
    print(f"Redundancia Hamming %: {(removed / total * 100):.2f}%")
    print("-" * 80)

    print("Duplicados eliminables por fuente:")
    if removed_by_source.empty:
        print("  (sin datos)")
    else:
        for _, row in removed_by_source.iterrows():
            print(f"  - {row['source']}: {row['removed']}")
    print("-" * 80)

    print("Cantidad de clústeres por tipo:")
    for k, v in cluster_type_counts.items():
        print(f"  - {k}: {v}")
    print("-" * 80)

    print("Duplicados eliminables por tipo de clúster:")
    if removed_by_cluster_type.empty:
        print("  (sin datos)")
    else:
        for _, row in removed_by_cluster_type.iterrows():
            print(f"  - {row['cluster_type']}: {row['removed']}")
    print("=" * 80)

    out_csv = config.EXPORTS_DIR / "hamming_source_report.csv"
    removed_by_source.to_csv(out_csv, index=False)
    print(f"CSV guardado en: {out_csv}")


if __name__ == "__main__":
    main()
