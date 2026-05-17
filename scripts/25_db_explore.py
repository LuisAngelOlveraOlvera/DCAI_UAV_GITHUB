#!/usr/bin/env python3
"""

          SQLite DB Explorer — Estructura & Trazabilidad              
                                                                      
  Bases soportadas:                                                   
    • analysis_metadata.sqlite   (análisis, scores, dedup, labels)    
    • dataset_master.sqlite      (imágenes maestras + anotaciones)    
                                                                      
  Uso:                                                                
    python db_explorer.py                        # directorio actual  
    python db_explorer.py --db ruta/archivo.db   # DB específica      
    python db_explorer.py --export               # genera HTML + CSV  
    python db_explorer.py --cross                # análisis cruzado   

"""

import sqlite3
import os
import sys
import json
import argparse
import csv
import hashlib
from datetime import datetime
from pathlib import Path

# 
#  CONFIGURACIÓN
# 

DEFAULT_DBS = [
    "analysis_metadata.sqlite",
    "dataset_master.sqlite",
]

SEP = "" * 72

# 
#  COLORES ANSI
# 

USE_COLOR = sys.stdout.isatty()

def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text

def H(t):  return _c(t, "1;36")   # cyan bold  → headers principales
def B(t):  return _c(t, "1")      # bold
def G(t):  return _c(t, "32")     # verde → ok / positivo
def Y(t):  return _c(t, "33")     # amarillo → advertencia
def R(t):  return _c(t, "31")     # rojo → error / crítico
def D(t):  return _c(t, "2")      # dim → texto secundario
def C(t):  return _c(t, "36")     # cyan → nombres de columnas


# 
#  METADATOS DEL ARCHIVO
# 

def file_meta(path: str) -> dict:
    p = Path(path)
    st = p.stat()
    with open(path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    return {
        "path":       str(p.resolve()),
        "size_bytes": st.st_size,
        "size_kb":    round(st.st_size / 1024, 2),
        "modified":   datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "md5":        md5,
    }


# 
#  HELPERS SQLITE
# 

def get_objects(conn, kind):
    return [r[0] for r in conn.execute(
        f"SELECT name FROM sqlite_master WHERE type='{kind}' ORDER BY name"
    ).fetchall()]


def get_col_info(conn, table) -> list:
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    return [{
        "cid":     r[0],
        "name":    r[1],
        "type":    r[2] or "TEXT",
        "notnull": bool(r[3]),
        "default": r[4],
        "pk":      bool(r[5]),
    } for r in rows]


def get_fks(conn, table) -> list:
    rows = conn.execute(f"PRAGMA foreign_key_list('{table}')").fetchall()
    return [{
        "from_col":  r[3],
        "ref_table": r[2],
        "to_col":    r[4],
        "on_update": r[5],
        "on_delete": r[6],
    } for r in rows]


def get_index_list(conn, table) -> list:
    idxs = conn.execute(f"PRAGMA index_list('{table}')").fetchall()
    result = []
    for idx in idxs:
        cols = [r[2] for r in conn.execute(f"PRAGMA index_info('{idx[1]}')").fetchall()]
        result.append({"name": idx[1], "unique": bool(idx[2]), "columns": cols})
    return result


def row_count(conn, table) -> int:
    try:
        return conn.execute(f"SELECT COUNT(*) FROM '{table}'").fetchone()[0]
    except Exception:
        return -1


def sample_rows(conn, table, n=3) -> list:
    try:
        cur = conn.execute(f"SELECT * FROM '{table}' LIMIT {n}")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception:
        return []


def distribution(conn, table, col, limit=10) -> list:
    try:
        return conn.execute(
            f"SELECT {col}, COUNT(*) as n FROM '{table}' "
            f"GROUP BY {col} ORDER BY n DESC LIMIT {limit}"
        ).fetchall()
    except Exception:
        return []


def numeric_stats(conn, table, col):
    try:
        r = conn.execute(
            f"SELECT MIN({col}), MAX({col}), AVG({col}), COUNT({col}) "
            f"FROM '{table}' WHERE {col} IS NOT NULL"
        ).fetchone()
        return {"min": r[0], "max": r[1], "avg": r[2], "count": r[3]}
    except Exception:
        return None


# 
#  IMPRESIÓN DE ESTRUCTURA DE TABLA
# 

def print_table(conn, table):
    cnt  = row_count(conn, table)
    cols = get_col_info(conn, table)
    fks  = get_fks(conn, table)
    idxs = get_index_list(conn, table)

    label = G(f"{cnt:,} filas") if cnt >= 0 else R("error")
    print(f"\n  {SEP}")
    print(f"  {B('Tabla:')} {C(table)}  {D('(' + str(cnt) + ' filas)')}")

    # Columnas
    print(D(f"\n  {'cid':<4} {'nombre':<26} {'tipo':<14} {'PK':^5} {'NN':^4} {'default'}"))
    print(D("  " + "" * 65))
    for c in cols:
        pk   = G(" PK") if c["pk"]      else "     "
        nn   = Y("NN")   if c["notnull"] else "  "
        dflt = str(c["default"]) if c["default"] is not None else ""
        print(f"  {str(c['cid']):<4} {C(c['name']):<34} {c['type']:<14} {pk:^10} {nn:^6} {D(dflt)}")

    # Foreign Keys
    if fks:
        print(B("\n   Foreign Keys:"))
        for fk in fks:
            print(
                f"    {C(fk['from_col'])} → {fk['ref_table']}.{C(fk['to_col'])}"
                f"  {D('ON UPDATE ' + fk['on_update'] + ' | ON DELETE ' + fk['on_delete'])}"
            )

    # Índices
    if idxs:
        print(B("\n    Índices:"))
        for idx in idxs:
            u = G("[UNIQUE]") if idx["unique"] else ""
            print(f"    {idx['name']}  {u}  →  {', '.join(idx['columns'])}")

    # Muestra
    samples = sample_rows(conn, table, 3)
    if samples:
        print(B("\n   Muestra (3 filas):"))
        for i, row in enumerate(samples, 1):
            preview = {k: (str(v)[:70] + "…" if v and len(str(v)) > 70 else v)
                       for k, v in row.items()}
            print(f"    [{i}] {json.dumps(preview, ensure_ascii=False, default=str)}")


# 
#  ESTADÍSTICAS ESPECÍFICAS: analysis_metadata.sqlite
# 

def stats_analysis_metadata(conn):
    """
    Estadísticas de trazabilidad para:
      Tabla: analysis_images
      Columnas: source | label_issue | processed | is_dedup_keep |
                judge_has_pred | judge_score | judge_max_iou | judge_pred_count
    """
    print(f"\n  {SEP}")
    print(H("   ESTADÍSTICAS DE TRAZABILIDAD — analysis_metadata"))

    total = row_count(conn, "analysis_images")

    # Fuente de los datos
    print(B("\n   Origen de imágenes (source):"))
    for val, cnt in distribution(conn, "analysis_images", "source"):
        bar = "" * max(1, int(cnt / 500))
        pct = cnt / total * 100
        print(f"    {str(val):<14} {cnt:>7,}  {bar}  {pct:.1f}%")

    # Calidad de etiquetas
    print(B("\n    Calidad de etiquetas (label_issue):"))
    color_map = {"ok": G, "bad_label": R, "poor_alignment": Y, "missing_label": R}
    for val, cnt in distribution(conn, "analysis_images", "label_issue"):
        fn  = color_map.get(str(val), D)
        pct = cnt / total * 100
        print(f"    {fn(str(val)):<30}  {cnt:>7,}  {pct:.2f}%")

    # Procesamiento completado
    proc = dict(distribution(conn, "analysis_images", "processed"))
    print(B("\n    Procesamiento (processed):"))
    print(f"    Procesadas   : {G(str(proc.get(1, 0)))} / {total}")
    if proc.get(0, 0):
        print(f"    Sin procesar : {Y(str(proc.get(0, 0)))}")

    # Deduplicación
    dedup = dict(distribution(conn, "analysis_images", "is_dedup_keep"))
    keep  = dedup.get(1, 0)
    duped = dedup.get(0, 0)
    clust = conn.execute("SELECT COUNT(DISTINCT dedup_cluster) FROM analysis_images").fetchone()[0]
    print(B("\n   Deduplicación:"))
    print(f"    Imágenes a mantener (is_dedup_keep=1) : {G(str(keep))}")
    print(f"    Duplicados a excluir (is_dedup_keep=0): {R(str(duped))}")
    print(f"    Clusters únicos de dedup               : {clust}")

    # Judge predictions
    pred = dict(distribution(conn, "analysis_images", "judge_has_pred"))
    print(B("\n   Predicciones del judge (judge_has_pred):"))
    print(f"    Con predicción    (=1): {G(str(pred.get(1, 0)))}")
    print(f"    Sin predicción    (=0): {Y(str(pred.get(0, 0)))}")

    # judge_score
    s = numeric_stats(conn, "analysis_images", "judge_score")
    if s:
        print(B("\n   judge_score  (calidad de predicción, 0 – 1):"))
        print(f"    mín={s['min']:.4f}  máx={s['max']:.4f}  promedio={s['avg']:.4f}  n={s['count']:,}")

    # judge_max_iou
    s = numeric_stats(conn, "analysis_images", "judge_max_iou")
    if s:
        print(B("\n   judge_max_iou  (solapamiento bbox anotación/predicción, 0 – 1):"))
        print(f"    mín={s['min']:.4f}  máx={s['max']:.4f}  promedio={s['avg']:.4f}")

    # judge_pred_count
    s = numeric_stats(conn, "analysis_images", "judge_pred_count")
    if s:
        print(B("\n   judge_pred_count  (número de predicciones por imagen):"))
        print(f"    mín={int(s['min'])}  máx={int(s['max'])}  promedio={s['avg']:.2f}")

    # Tabla resumen rápido
    print(B("\n   Resumen ejecutivo:"))
    print(f"    Total imágenes analizadas : {total:,}")
    lbl_ok    = conn.execute("SELECT COUNT(*) FROM analysis_images WHERE label_issue='ok'").fetchone()[0]
    lbl_bad   = total - lbl_ok
    score_ok  = conn.execute("SELECT COUNT(*) FROM analysis_images WHERE judge_score >= 0.8").fetchone()[0]
    iou_ok    = conn.execute("SELECT COUNT(*) FROM analysis_images WHERE judge_max_iou >= 0.5").fetchone()[0]
    print(f"    label_issue = ok           : {G(str(lbl_ok))} ({lbl_ok/total*100:.1f}%)")
    print(f"    label_issue ≠ ok (revisar) : {Y(str(lbl_bad))} ({lbl_bad/total*100:.1f}%)")
    print(f"    judge_score ≥ 0.80         : {G(str(score_ok))} ({score_ok/total*100:.1f}%)")
    print(f"    judge_max_iou ≥ 0.50       : {G(str(iou_ok))} ({iou_ok/total*100:.1f}%)")


# 
#  ESTADÍSTICAS ESPECÍFICAS: dataset_master.sqlite
# 

def stats_dataset_master(conn):
    """
    Estadísticas de trazabilidad para:
      Tabla images      : id | file_name | relative_path | width | height | phash | label_issue
      Tabla annotations : id | image_id (FK) | class_id | x_center | y_center | width | height
    """
    print(f"\n  {SEP}")
    print(H("   ESTADÍSTICAS DE TRAZABILIDAD — dataset_master"))

    n_img = row_count(conn, "images")
    n_ann = row_count(conn, "annotations")

    # Resumen general
    print(B("\n   Resumen general:"))
    print(f"    Imágenes en catálogo  : {G(str(n_img))}")
    print(f"    Anotaciones totales   : {G(str(n_ann))}")
    print(f"    Promedio ann/imagen   : {n_ann/n_img:.2f}" if n_img else "")

    no_ann = conn.execute(
        "SELECT COUNT(*) FROM images WHERE id NOT IN "
        "(SELECT DISTINCT image_id FROM annotations)"
    ).fetchone()[0]
    icon = R(str(no_ann)) if no_ann else G("0")
    print(f"    Imágenes sin anotación: {icon}  {'  requiere revisión' if no_ann else ''}")

    # Distribución de clases
    print(B("\n    Distribución de clases (annotations.class_id):"))
    for val, cnt in distribution(conn, "annotations", "class_id"):
        pct = cnt / n_ann * 100
        print(f"    class_id={val}  →  {G(f'{cnt:,}'):>9}  ({pct:.1f}%)")

    # Calidad de etiquetas
    print(B("\n    Calidad de etiquetas (images.label_issue):"))
    color_map = {"ok": G, "bad_label": R, "poor_alignment": Y, "missing_label": R}
    for val, cnt in distribution(conn, "images", "label_issue"):
        fn  = color_map.get(str(val), D)
        pct = cnt / n_img * 100
        print(f"    {fn(str(val)):<30}  {cnt:>7,}  {pct:.2f}%")

    # Anotaciones por imagen
    s = conn.execute(
        "SELECT MIN(n), MAX(n), AVG(n) FROM "
        "(SELECT image_id, COUNT(*) as n FROM annotations GROUP BY image_id)"
    ).fetchone()
    print(B("\n   Anotaciones por imagen:"))
    print(f"    mín={int(s[0])}  máx={int(s[1])}  promedio={s[2]:.2f}")

    # Top imágenes más anotadas
    print(B("\n   Imágenes con más anotaciones (top 5):"))
    top = conn.execute(
        "SELECT i.file_name, COUNT(a.id) as n "
        "FROM annotations a JOIN images i ON a.image_id=i.id "
        "GROUP BY a.image_id ORDER BY n DESC LIMIT 5"
    ).fetchall()
    for row in top:
        print(f"    {row[0]:<40}  {row[1]} anotaciones")

    # Dimensiones de imagen
    sw = numeric_stats(conn, "images", "width")
    sh = numeric_stats(conn, "images", "height")
    if sw and sh:
        print(B("\n    Dimensiones de imágenes (píxeles):"))
        print(f"    width  — mín={int(sw['min'])}  máx={int(sw['max'])}  promedio={sw['avg']:.0f}")
        print(f"    height — mín={int(sh['min'])}  máx={int(sh['max'])}  promedio={sh['avg']:.0f}")

    # Bounding boxes normalizadas
    print(B("\n   Bounding boxes normalizadas (valores 0 – 1):"))
    for col in ("x_center", "y_center", "width", "height"):
        s = numeric_stats(conn, "annotations", col)
        if s:
            print(f"    {col:<10}  mín={s['min']:.4f}  máx={s['max']:.4f}  promedio={s['avg']:.4f}")

    # Resumen ejecutivo
    print(B("\n   Resumen ejecutivo:"))
    lbl_ok  = conn.execute("SELECT COUNT(*) FROM images WHERE label_issue='ok'").fetchone()[0]
    lbl_bad = n_img - lbl_ok
    print(f"    Total imágenes en catálogo : {n_img:,}")
    print(f"    Total anotaciones          : {n_ann:,}")
    print(f"    label_issue = ok           : {G(str(lbl_ok))} ({lbl_ok/n_img*100:.1f}%)")
    print(f"    label_issue ≠ ok           : {Y(str(lbl_bad))} ({lbl_bad/n_img*100:.1f}%)")
    print(f"    Imágenes sin anotaciones   : {(R if no_ann else G)(str(no_ann))}")


# 
#  ANÁLISIS CRUZADO DE TRAZABILIDAD ENTRE AMBAS DBs
# 

def cross_analysis(path_am: str, path_dm: str):
    """
    Cruza analysis_metadata.sqlite ↔ dataset_master.sqlite
    Claves de trazabilidad: id, phash, relative_path, label_issue
    """
    print(f"\n{'' * 72}")
    print(H("   ANÁLISIS CRUZADO DE TRAZABILIDAD"))
    print(f"  {D(Path(path_am).name)}  ↔  {D(Path(path_dm).name)}")
    print(f"{'' * 72}")

    conn_am = sqlite3.connect(path_am)
    conn_dm = sqlite3.connect(path_dm)

    n_am = row_count(conn_am, "analysis_images")
    n_dm = row_count(conn_dm, "images")
    n_ann = row_count(conn_dm, "annotations")

    # 1. Conteos
    print(B("\n  1. Conteos totales:"))
    print(f"     analysis_images   : {n_am:>8,}")
    print(f"     images (master)   : {n_dm:>8,}")
    match_icon = G(" COINCIDEN") if n_am == n_dm else Y(f"≠ DIFIEREN  (delta: {abs(n_am-n_dm)})")
    print(f"     ¿Mismo total?     : {match_icon}")
    print(f"     annotations       : {n_ann:>8,}")

    # 2. Integridad de join id + phash
    print(B("\n  2. Integridad del join por (id, phash):"))
    conn_am.execute("ATTACH DATABASE ? AS master", (path_dm,))
    try:
        mismatch = conn_am.execute(
            "SELECT COUNT(*) FROM analysis_images a "
            "LEFT JOIN master.images b ON a.id = b.id AND a.phash = b.phash "
            "WHERE b.id IS NULL"
        ).fetchone()[0]
        linked = n_am - mismatch
        print(f"     Con match (id+phash)  : {G(f'{linked:,}'):>10}")
        icon   = G(" ÍNTEGRO") if mismatch == 0 else R(f" {mismatch} sin match")
        print(f"     Sin match             : {(R if mismatch else G)(str(mismatch)):>10}  {icon}")
    except Exception as e:
        print(f"     {R(f'Error en join cruzado: {e}')}")
    finally:
        try:
            conn_am.execute("DETACH DATABASE master")
        except Exception:
            pass

    # 3. Consistencia de label_issue
    print(B("\n  3. Consistencia de label_issue entre DBs:"))
    dist_am = dict(conn_am.execute(
        "SELECT label_issue, COUNT(*) FROM analysis_images GROUP BY label_issue"
    ).fetchall())
    dist_dm = dict(conn_dm.execute(
        "SELECT label_issue, COUNT(*) FROM images GROUP BY label_issue"
    ).fetchall())
    all_keys = sorted(set(list(dist_am.keys()) + list(dist_dm.keys())))
    print(f"     {'label_issue':<24} {'analysis_metadata':>17}  {'dataset_master':>14}  {'ok?':>5}")
    print(f"     {D(''*65)}")
    for k in all_keys:
        v1 = dist_am.get(k, 0)
        v2 = dist_dm.get(k, 0)
        m  = G("") if v1 == v2 else R("")
        print(f"     {str(k):<24} {v1:>17,}  {v2:>14,}  {m:>5}")

    # 4. Duplicados marcados vs anotaciones activas
    print(B("\n  4. Duplicados (is_dedup_keep=0) con anotaciones en dataset_master:"))
    dup_ids = [r[0] for r in conn_am.execute(
        "SELECT id FROM analysis_images WHERE is_dedup_keep=0"
    ).fetchall()]
    print(f"     Total marcados como duplicado: {len(dup_ids)}")
    if dup_ids:
        placeholders = ",".join("?" * len(dup_ids))
        still_ann = conn_dm.execute(
            f"SELECT COUNT(DISTINCT image_id) FROM annotations "
            f"WHERE image_id IN ({placeholders})", dup_ids
        ).fetchone()[0]
        icon = Y("  requieren revisión") if still_ann else G(" sin anotaciones activas")
        print(f"     Duplicados con anotaciones   : {still_ann}  {icon}")

    # 5. Imágenes con problemas de etiqueta y sus anotaciones
    print(B("\n  5. Imágenes con label_issue ≠ ok y sus anotaciones:"))
    bad_ids = [r[0] for r in conn_am.execute(
        "SELECT id FROM analysis_images WHERE label_issue != 'ok'"
    ).fetchall()]
    print(f"     Imágenes con problemas: {len(bad_ids)}")
    if bad_ids:
        placeholders = ",".join("?" * len(bad_ids))
        ann_bad = conn_dm.execute(
            f"SELECT COUNT(*) FROM annotations WHERE image_id IN ({placeholders})",
            bad_ids
        ).fetchone()[0]
        print(f"     Anotaciones asociadas : {Y(str(ann_bad))}  (evaluar si deben excluirse del entrenamiento)")

    # 6. Imágenes con judge_score bajo (< 0.5) que siguen en el master
    print(B("\n  6. Imágenes con judge_score < 0.5 (baja calidad de predicción):"))
    low_score = [r[0] for r in conn_am.execute(
        "SELECT id FROM analysis_images WHERE judge_score < 0.5 AND judge_has_pred=1"
    ).fetchall()]
    print(f"     Con score < 0.5       : {len(low_score)}")
    if low_score:
        placeholders = ",".join("?" * len(low_score))
        ann_low = conn_dm.execute(
            f"SELECT COUNT(*) FROM annotations WHERE image_id IN ({placeholders})",
            low_score
        ).fetchone()[0]
        print(f"     Anotaciones asociadas : {Y(str(ann_low))}")

    conn_am.close()
    conn_dm.close()


# 
#  EXPORTACIÓN HTML
# 

def export_html(db_path: str, tables_data: list) -> str:
    name = Path(db_path).stem
    meta = file_meta(db_path)

    blocks = ""
    for t in tables_data:
        col_rows = "".join(
            f"<tr><td>{c['cid']}</td><td><b>{c['name']}</b></td><td>{c['type']}</td>"
            f"<td style='color:#4ade80'>{'' if c['pk'] else ''}</td>"
            f"<td style='color:#facc15'>{'!' if c['notnull'] else ''}</td>"
            f"<td style='color:#94a3b8'>{c['default'] or ''}</td></tr>"
            for c in t["cols"]
        )
        fk_block = ""
        if t["fks"]:
            fk_rows = "".join(
                f"<tr><td>{fk['from_col']}</td><td>→</td>"
                f"<td>{fk['ref_table']}.{fk['to_col']}</td>"
                f"<td style='color:#94a3b8'>{fk['on_delete']}</td></tr>"
                for fk in t["fks"]
            )
            fk_block = (
                "<h4 style='color:#94a3b8;margin-top:1rem'> Foreign Keys</h4>"
                "<table><tr><th>desde</th><th></th><th>hacia</th><th>ON DELETE</th></tr>"
                f"{fk_rows}</table>"
            )
        blocks += f"""
        <div class='tbl'>
          <h3> {t['name']} <span class='badge'>{t['rows']:,} filas</span></h3>
          <table>
            <tr><th>cid</th><th>Columna</th><th>Tipo</th><th>PK</th><th>NN</th><th>Default</th></tr>
            {col_rows}
          </table>
          {fk_block}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>DB Explorer — {name}</title>
  <style>
    *    {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ font-family:'Segoe UI',system-ui,monospace; background:#0b0f1a;
           color:#e2e8f0; padding:2.5rem; line-height:1.5; }}
    h1   {{ color:#67e8f9; font-size:1.6rem; border-bottom:2px solid #1e293b;
           padding-bottom:.75rem; margin-bottom:1.5rem; }}
    h2   {{ color:#38bdf8; font-size:1.2rem; margin:2rem 0 .75rem; }}
    h3   {{ color:#a5f3fc; font-size:1rem; margin-bottom:.75rem; }}
    h4   {{ color:#94a3b8; font-size:.9rem; }}
    .meta {{ background:#111827; border:1px solid #1e293b; border-radius:8px;
             padding:1rem 1.5rem; margin-bottom:1.5rem; font-size:.83rem;
             color:#94a3b8; line-height:1.8; }}
    .meta b {{ color:#cbd5e1; }}
    .badge {{ background:#1e3a5f; color:#7dd3fc; border-radius:4px;
              font-size:.72rem; padding:2px 8px; margin-left:8px;
              vertical-align:middle; }}
    .tbl  {{ background:#0d1526; border:1px solid #1e293b; border-radius:10px;
             padding:1.25rem 1.75rem; margin:1rem 0; }}
    table {{ border-collapse:collapse; width:100%; margin:.5rem 0; }}
    th    {{ background:#1e293b; color:#7dd3fc; text-align:left;
             padding:7px 12px; font-size:.82rem; font-weight:600; }}
    td    {{ border-bottom:1px solid #151f2e; padding:6px 12px; font-size:.83rem; }}
    tr:hover td {{ background:#0f1c30; }}
  </style>
</head>
<body>
  <h1>  DB Explorer — {name}</h1>
  <div class="meta">
    <b>Ruta:</b> {meta['path']}<br>
    <b>Tamaño:</b> {meta['size_kb']} KB &nbsp;({meta['size_bytes']:,} bytes)
    &nbsp;|&nbsp; <b>Modificado:</b> {meta['modified']}
    &nbsp;|&nbsp; <b>MD5:</b> {meta['md5']}
  </div>
  <h2>Tablas</h2>
  {blocks}
</body>
</html>"""

    out = Path(db_path).stem + "_explorer.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


# 
#  EXPORTACIÓN CSV DEL SCHEMA
# 

def export_csv(db_path: str, conn, tables: list) -> str:
    out = Path(db_path).stem + "_schema.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["db", "table", "cid", "column", "type", "pk", "notnull", "default", "row_count"])
        for t in tables:
            cnt  = row_count(conn, t)
            cols = get_col_info(conn, t)
            for c in cols:
                w.writerow([
                    Path(db_path).name, t, c["cid"], c["name"],
                    c["type"], int(c["pk"]), int(c["notnull"]),
                    c["default"] or "", cnt
                ])
    return out


# 
#  ANÁLISIS COMPLETO DE UNA DB
# 

def analyze_db(db_path: str, do_export: bool = False):
    if not os.path.exists(db_path):
        print(R(f"\n   Archivo no encontrado: {db_path}"))
        return

    meta = file_meta(db_path)
    conn = sqlite3.connect(db_path)

    print(f"\n{'' * 72}")
    print(H(f"   BASE DE DATOS: {Path(db_path).name}"))
    print(f"{'' * 72}")

    # Metadatos del archivo
    print(B("\n   Archivo"))
    print(f"     Ruta       : {meta['path']}")
    print(f"     Tamaño     : {meta['size_kb']} KB  ({meta['size_bytes']:,} bytes)")
    print(f"     Modificado : {meta['modified']}")
    print(f"     MD5        : {D(meta['md5'])}")

    # Info SQLite engine
    psize  = conn.execute("PRAGMA page_size").fetchone()[0]
    pcount = conn.execute("PRAGMA page_count").fetchone()[0]
    freep  = conn.execute("PRAGMA freelist_count").fetchone()[0]
    integ  = conn.execute("PRAGMA integrity_check").fetchone()[0]
    ver    = conn.execute("SELECT sqlite_version()").fetchone()[0]

    print(B("\n   SQLite Engine"))
    print(f"     Versión         : {ver}")
    print(f"     Page size       : {psize} bytes")
    print(f"     Páginas totales : {pcount}  (≈ {pcount * psize // 1024} KB ocupados)")
    print(f"     Páginas libres  : {freep}")
    print(f"     Integridad      : {G(' OK') if integ == 'ok' else R(' ' + integ)}")

    # Objetos de la DB
    tables   = get_objects(conn, "table")
    views    = get_objects(conn, "view")
    triggers = get_objects(conn, "trigger")
    idx_all  = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
    ).fetchall()

    user_tables = [t for t in tables if not t.startswith("sqlite_")]

    print(B(f"\n    Objetos"))
    print(f"     Tablas   : {len(user_tables)}  {D(str(user_tables))}")
    print(f"     Vistas   : {len(views)}" + (f"  {D(str(views))}" if views else ""))
    print(f"     Índices  : {len(idx_all)}")
    print(f"     Triggers : {len(triggers)}")

    # Estructura de cada tabla
    tables_data = []
    for table in user_tables:
        print_table(conn, table)
        tables_data.append({
            "name": table,
            "rows": row_count(conn, table),
            "cols": get_col_info(conn, table),
            "fks":  get_fks(conn, table),
        })

    # Estadísticas específicas por DB
    db_name = Path(db_path).name
    if db_name == "analysis_metadata.sqlite" and "analysis_images" in user_tables:
        stats_analysis_metadata(conn)
    elif db_name == "dataset_master.sqlite" and "images" in user_tables:
        stats_dataset_master(conn)

    # Exportar si se pidió
    if do_export:
        html_out = export_html(db_path, tables_data)
        csv_out  = export_csv(db_path, conn, user_tables)
        print(f"\n  {G(' HTML exportado:')} {html_out}")
        print(f"  {G(' CSV exportado :')} {csv_out}")

    conn.close()


# 
#  ENTRY POINT
# 

def main():
    parser = argparse.ArgumentParser(
        description="Explorador de estructura y trazabilidad de SQLite DBs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python db_explorer.py
  python db_explorer.py --db ./data/analysis_metadata.sqlite ./data/dataset_master.sqlite
  python db_explorer.py --export
  python db_explorer.py --cross
        """
    )
    parser.add_argument("--db",     nargs="+",        help="Rutas a archivos .sqlite (máx 2 para análisis cruzado)")
    parser.add_argument("--export", action="store_true", help="Exportar reporte HTML y CSV del schema")
    parser.add_argument("--cross",  action="store_true", help="Ejecutar análisis cruzado entre las dos DBs")
    args = parser.parse_args()

    db_paths = args.db if args.db else DEFAULT_DBS

    found = [p for p in db_paths if os.path.exists(p)]
    if not found:
        print(Y("\n    No se encontraron las DBs en el directorio actual."))
        print(f"  Usa --db <ruta> o coloca los archivos junto a este script.\n")
        for p in db_paths:
            print(f"    {R('')} {p}")
        sys.exit(1)

    for db_path in db_paths:
        if os.path.exists(db_path):
            analyze_db(db_path, do_export=args.export)
        else:
            print(Y(f"\n    Saltando (no encontrada): {db_path}"))

    # Análisis cruzado: automático si están ambas DBs, o con --cross
    am = next((p for p in db_paths if "analysis_metadata" in p and os.path.exists(p)), None)
    dm = next((p for p in db_paths if "dataset_master"    in p and os.path.exists(p)), None)
    if am and dm and (args.cross or True):
        cross_analysis(am, dm)

    print(f"\n{'' * 72}")
    print(G("   Análisis completado."))
    print(f"{'' * 72}\n")


if __name__ == "__main__":
    main()