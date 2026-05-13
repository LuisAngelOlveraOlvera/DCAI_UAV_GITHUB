"""
05 DATASET INTEGRITY VALIDATION (portable dataset)
-----------------------------------------------------------------------
Validate dataset consistency by:
1) checking that indexed images exist on disk
2) checking that annotation text files exist
3) reporting missing files and integrity issues
-----------------------------------------------------------------------
"""

import sqlite3
from pathlib import Path
import config

def validate():
    print(" VALIDANDO INTEGRIDAD DEL DATASET PORTABLE\n")

    conn = sqlite3.connect(config.DB_PATH)
    cur = conn.cursor()

    errors = 0

    cur.execute("SELECT id, relative_path, width, height FROM images")
    images = cur.fetchall()

    for img_id, rel_path, w, h in images:
        img_path = config.DATASET_ROOT / rel_path

        if not img_path.exists():
            print(f" Imagen faltante: {rel_path}")
            errors += 1
            continue

        from PIL import Image
        with Image.open(img_path) as im:
            iw, ih = im.size
            if iw != w or ih != h:
                print(f" Dimensión inconsistente: {rel_path}")
                errors += 1

        cur.execute("SELECT x_center,y_center,width,height FROM annotations WHERE image_id=?", (img_id,))
        annos = cur.fetchall()

        if not annos:
            print(f" Imagen sin anotaciones: {rel_path}")

        for xc, yc, bw, bh in annos:
            if not all(0 <= v <= 1 for v in [xc, yc, bw, bh]):
                print(f" Bounding box fuera de rango [0,1]: {rel_path}")
                errors += 1

    cur.execute("""
        SELECT COUNT(*) FROM annotations
        WHERE image_id NOT IN (SELECT id FROM images)
    """)
    orphan_ann = cur.fetchone()[0]
    if orphan_ann > 0:
        print(f" Anotaciones huérfanas: {orphan_ann}")
        errors += orphan_ann

    conn.close()

    if errors == 0:
        print("\n DATASET ÍNTEGRO Y PORTABLE")
    else:
        print(f"\n ERRORES DETECTADOS: {errors}")

if __name__ == "__main__":
    validate()
