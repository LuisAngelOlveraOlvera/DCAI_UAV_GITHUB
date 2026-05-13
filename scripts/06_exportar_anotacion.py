"""
06 ANNOTATION EXPORT (multi-format)
-----------------------------------------------------------------------
Export annotations from dataset_master.sqlite to:
1) YOLO
2) COCO
3) Pascal VOC
4) TensorFlow metadata JSON
5) PyTorch, CSV, Detectron2, and CreateML
-----------------------------------------------------------------------
"""

import sqlite3
import json
import csv
import pickle
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree
import config
import utils

# Configure logger
logger = utils.setup_logger("ExportAnnotations", 
                           log_file=str(config.LOGS_DIR / "export_annotations.log"))

# ======================================================
# EXPORT DIRECTORIES
# ======================================================
EXPORT_DIR = config.EXPORTS_DIR
YOLO_DIR = EXPORT_DIR / "yolo"
COCO_DIR = EXPORT_DIR / "coco"
VOC_DIR = EXPORT_DIR / "pascal_voc"
TF_DIR = EXPORT_DIR / "tensorflow"
PYTORCH_DIR = EXPORT_DIR / "pytorch"
CSV_DIR = EXPORT_DIR / "csv"
DETECTRON_DIR = EXPORT_DIR / "detectron2"
CREATEML_DIR = EXPORT_DIR / "createml"

for d in [YOLO_DIR, COCO_DIR, VOC_DIR, TF_DIR, PYTORCH_DIR, 
          CSV_DIR, DETECTRON_DIR, CREATEML_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ======================================================
# DATABASE CONNECTION
# ======================================================
conn = sqlite3.connect(config.DB_PATH)
cur = conn.cursor()

logger.info("=== INICIANDO EXPORTACIÓN MULTI-FORMATO ===")

# ======================================================
# 1. FORMATO YOLO (TXT) - NATIVO
# ======================================================
logger.info("Exportando a formato YOLO...")
cur.execute("SELECT id, file_name, relative_path, width, height FROM images")
images = cur.fetchall()

for img_id, fname, rpath, w, h in images:
    yolo_file = YOLO_DIR / f"{Path(fname).stem}.txt"
    
    cur.execute("""
        SELECT class_id, x_center, y_center, width, height
        FROM annotations WHERE image_id=?
    """, (img_id,))
    
    with open(yolo_file, "w") as f:
        for cls, xc, yc, bw, bh in cur.fetchall():
            f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

logger.info(f" YOLO: {len(images)} archivos .txt generados")

# ======================================================
# 2. FORMATO COCO (JSON)
# ======================================================
logger.info("Exportando a formato COCO...")
coco = {
    "info": {
        "description": "DATASET_TESIS - Person Detection",
        "version": "1.0",
        "year": 2025,
        "contributor": "DCAI Methodology Validation",
        "date_created": "2025-12-15"
    },
    "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
    "images": [],
    "annotations": [],
    "categories": [{"id": 0, "name": "person", "supercategory": "person"}]
}

ann_id = 1

cur.execute("SELECT id, file_name, relative_path, width, height FROM images")
images = cur.fetchall()

for img_id, fname, rpath, w, h in images:
    coco["images"].append({
        "id": img_id,
        "file_name": rpath,
        "width": w,
        "height": h,
        "license": 1,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": ""
    })

    cur.execute("""
        SELECT class_id, x_center, y_center, width, height
        FROM annotations WHERE image_id=?
    """, (img_id,))

    for cls, xc, yc, bw, bh in cur.fetchall():
        x = (xc - bw/2) * w
        y = (yc - bh/2) * h
        bw_px = bw * w
        bh_px = bh * h

        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": int(cls),
            "bbox": [x, y, bw_px, bh_px],
            "area": bw_px * bh_px,
            "iscrowd": 0,
            "segmentation": []
        })
        ann_id += 1

with open(COCO_DIR / "annotations.json", "w") as f:
    json.dump(coco, f, indent=2)

logger.info(f" COCO: {len(coco['images'])} imágenes, {len(coco['annotations'])} anotaciones")

# ======================================================
# 3. FORMATO PASCAL VOC (XML)
# ======================================================
logger.info("Exportando a formato Pascal VOC...")
voc_count = 0

for img_id, fname, rpath, w, h in images:
    ann = Element("annotation")

    SubElement(ann, "folder").text = "images"
    SubElement(ann, "filename").text = fname
    SubElement(ann, "path").text = rpath

    source = SubElement(ann, "source")
    SubElement(source, "database").text = "DATASET_TESIS"

    size = SubElement(ann, "size")
    SubElement(size, "width").text = str(w)
    SubElement(size, "height").text = str(h)
    SubElement(size, "depth").text = "3"

    SubElement(ann, "segmented").text = "0"

    cur.execute("""
        SELECT class_id, x_center, y_center, width, height
        FROM annotations WHERE image_id=?
    """, (img_id,))

    for cls, xc, yc, bw, bh in cur.fetchall():
        obj = SubElement(ann, "object")
        SubElement(obj, "name").text = "person"
        SubElement(obj, "pose").text = "Unspecified"
        SubElement(obj, "truncated").text = "0"
        SubElement(obj, "difficult").text = "0"

        bnd = SubElement(obj, "bndbox")
        xmin = int((xc - bw/2) * w)
        ymin = int((yc - bh/2) * h)
        xmax = int((xc + bw/2) * w)
        ymax = int((yc + bh/2) * h)

        SubElement(bnd, "xmin").text = str(max(0, xmin))
        SubElement(bnd, "ymin").text = str(max(0, ymin))
        SubElement(bnd, "xmax").text = str(min(w, xmax))
        SubElement(bnd, "ymax").text = str(min(h, ymax))

    tree = ElementTree(ann)
    tree.write(VOC_DIR / f"{Path(fname).stem}.xml", encoding="utf-8", xml_declaration=True)
    voc_count += 1

logger.info(f" Pascal VOC: {voc_count} archivos .xml generados")

# ======================================================
# 4. FORMATO TENSORFLOW (TFRECORD) - METADATA JSON
# ======================================================
logger.info("Exportando metadatos para TensorFlow...")
# Note: TFRecord requires additional code for binary serialization
# Here we generate an intermediate JSON compatible with the TF Object Detection API

tf_data = {
    "dataset_name": "DATASET_TESIS_Person_Detection",
    "num_classes": 1,
    "class_names": ["person"],
    "images": []
}

for img_id, fname, rpath, w, h in images:
    img_data = {
        "filename": rpath,
        "width": w,
        "height": h,
        "objects": []
    }
    
    cur.execute("""
        SELECT class_id, x_center, y_center, width, height
        FROM annotations WHERE image_id=?
    """, (img_id,))
    
    for cls, xc, yc, bw, bh in cur.fetchall():
        xmin = (xc - bw/2) * w
        ymin = (yc - bh/2) * h
        xmax = (xc + bw/2) * w
        ymax = (yc + bh/2) * h
        
        img_data["objects"].append({
            "class": "person",
            "class_id": int(cls),
            "bbox": {
                "xmin": xmin / w,  # Normalized [0,1]
                "ymin": ymin / h,
                "xmax": xmax / w,
                "ymax": ymax / h
            }
        })
    
    tf_data["images"].append(img_data)

with open(TF_DIR / "dataset_metadata.json", "w") as f:
    json.dump(tf_data, f, indent=2)

logger.info(f" TensorFlow: Metadata JSON generado ({len(tf_data['images'])} imágenes)")
logger.info("  (Para TFRecord usar: tensorflow/object_detection/dataset_tools/create_coco_tf_record.py)")

# ======================================================
# 5. FORMATO PYTORCH (PICKLE)
# ======================================================
logger.info("Exportando a formato PyTorch...")
pytorch_data = {
    "images": [],
    "annotations": []
}

for img_id, fname, rpath, w, h in images:
    pytorch_data["images"].append({
        "id": img_id,
        "file_name": rpath,
        "width": w,
        "height": h
    })
    
    cur.execute("""
        SELECT class_id, x_center, y_center, width, height
        FROM annotations WHERE image_id=?
    """, (img_id,))
    
    boxes = []
    labels = []
    
    for cls, xc, yc, bw, bh in cur.fetchall():
        xmin = (xc - bw/2) * w
        ymin = (yc - bh/2) * h
        xmax = (xc + bw/2) * w
        ymax = (yc + bh/2) * h
        
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(int(cls))
    
    pytorch_data["annotations"].append({
        "image_id": img_id,
        "boxes": boxes,  # List[List[float]] in xyxy format
        "labels": labels,  # List[int]
        "area": [((b[2]-b[0])*(b[3]-b[1])) for b in boxes],
        "iscrowd": [0] * len(boxes)
    })

# Save as pickle (efficient for PyTorch)
with open(PYTORCH_DIR / "dataset.pkl", "wb") as f:
    pickle.dump(pytorch_data, f)

# Also save as JSON for inspection
with open(PYTORCH_DIR / "dataset.json", "w") as f:
    json.dump(pytorch_data, f, indent=2)

logger.info(f" PyTorch: {len(pytorch_data['images'])} imágenes serializadas")

# ======================================================
# 6. CSV FORMAT (PANDAS/ANALYSIS)
# ======================================================
logger.info("Exportando a formato CSV...")
csv_rows = []

for img_id, fname, rpath, w, h in images:
    cur.execute("""
        SELECT class_id, x_center, y_center, width, height
        FROM annotations WHERE image_id=?
    """, (img_id,))
    
    annotations = cur.fetchall()
    
    if len(annotations) == 0:
        # Image without annotations
        csv_rows.append({
            "image_id": img_id,
            "filename": fname,
            "relative_path": rpath,
            "width": w,
            "height": h,
            "class_id": None,
            "class_name": None,
            "xmin": None,
            "ymin": None,
            "xmax": None,
            "ymax": None,
            "bbox_width": None,
            "bbox_height": None,
            "bbox_area": None,
            "x_center_norm": None,
            "y_center_norm": None,
            "width_norm": None,
            "height_norm": None
        })
    else:
        for cls, xc, yc, bw, bh in annotations:
            xmin = int((xc - bw/2) * w)
            ymin = int((yc - bh/2) * h)
            xmax = int((xc + bw/2) * w)
            ymax = int((yc + bh/2) * h)
            
            csv_rows.append({
                "image_id": img_id,
                "filename": fname,
                "relative_path": rpath,
                "width": w,
                "height": h,
                "class_id": int(cls),
                "class_name": "person",
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "bbox_width": xmax - xmin,
                "bbox_height": ymax - ymin,
                "bbox_area": (xmax - xmin) * (ymax - ymin),
                "x_center_norm": xc,
                "y_center_norm": yc,
                "width_norm": bw,
                "height_norm": bh
            })

with open(CSV_DIR / "annotations.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_rows)

logger.info(f" CSV: {len(csv_rows)} filas exportadas")

# ======================================================
# 7. FORMATO DETECTRON2 (JSON)
# ======================================================
logger.info("Exportando a formato Detectron2...")
detectron_data = []

for img_id, fname, rpath, w, h in images:
    record = {
        "file_name": rpath,
        "image_id": img_id,
        "height": h,
        "width": w,
        "annotations": []
    }
    
    cur.execute("""
        SELECT class_id, x_center, y_center, width, height
        FROM annotations WHERE image_id=?
    """, (img_id,))
    
    for cls, xc, yc, bw, bh in cur.fetchall():
        xmin = (xc - bw/2) * w
        ymin = (yc - bh/2) * h
        box_w = bw * w
        box_h = bh * h
        
        record["annotations"].append({
            "bbox": [xmin, ymin, box_w, box_h],  # x, y, width, height
            "bbox_mode": 1,  # BoxMode.XYWH_ABS
            "category_id": int(cls),
            "iscrowd": 0
        })
    
    detectron_data.append(record)

with open(DETECTRON_DIR / "dataset_detectron2.json", "w") as f:
    json.dump(detectron_data, f, indent=2)

logger.info(f" Detectron2: {len(detectron_data)} registros generados")

# ======================================================
# 8. FORMATO CREATEML (JSON) - APPLE
# ======================================================
logger.info("Exportando a formato CreateML...")
createml_data = []

for img_id, fname, rpath, w, h in images:
    record = {
        "image": rpath,
        "annotations": []
    }
    
    cur.execute("""
        SELECT class_id, x_center, y_center, width, height
        FROM annotations WHERE image_id=?
    """, (img_id,))
    
    for cls, xc, yc, bw, bh in cur.fetchall():
        xmin = int((xc - bw/2) * w)
        ymin = int((yc - bh/2) * h)
        box_w = int(bw * w)
        box_h = int(bh * h)
        
        record["annotations"].append({
            "label": "person",
            "coordinates": {
                "x": xmin + box_w // 2,  # Absolute X center
                "y": ymin + box_h // 2,  # Absolute Y center
                "width": box_w,
                "height": box_h
            }
        })
    
    createml_data.append(record)

with open(CREATEML_DIR / "annotations_createml.json", "w") as f:
    json.dump(createml_data, f, indent=2)

logger.info(f" CreateML: {len(createml_data)} imágenes anotadas")

# ======================================================
# REPORTE FINAL
# ======================================================
conn.close()

logger.info("="*60)
logger.info("RESUMEN DE EXPORTACIÓN")
logger.info("="*60)
logger.info(f" YOLO:        {YOLO_DIR}")
logger.info(f" COCO:        {COCO_DIR}")
logger.info(f" Pascal VOC:  {VOC_DIR}")
logger.info(f" TensorFlow:  {TF_DIR}")
logger.info(f" PyTorch:     {PYTORCH_DIR}")
logger.info(f" CSV:         {CSV_DIR}")
logger.info(f" Detectron2:  {DETECTRON_DIR}")
logger.info(f" CreateML:    {CREATEML_DIR}")
logger.info("="*60)
logger.info(" EXPORTACIÓN MULTI-FORMATO COMPLETADA")
logger.info("="*60)

print("\n Todos los formatos exportados exitosamente!")
print(f" Directorio: {EXPORT_DIR}")
