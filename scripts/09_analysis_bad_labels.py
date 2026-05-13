"""
09 BAD LABEL INSPECTOR (interactive QA)
-----------------------------------------------------------------------
Inspect label-quality issues interactively by:
1) overlaying ground truth and judge predictions
2) adjusting the confidence threshold live
3) recording human findings and ROC-oriented review data
-----------------------------------------------------------------------
"""

import sqlite3
import cv2
import numpy as np
from pathlib import Path
import config
import utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

logger = utils.setup_logger("Visual_Inspector", log_file=str(config.LOGS_DIR / "visual_inspect.log"))

class InteractiveInspector:
    def __init__(self):
        self.current_threshold = config.CONF_THRESHOLD
        self.model = None
        self.findings = {"true_error": 0, "false_alarm": 0, "ambiguous": 0}
        self.roc_scores = []   # judge confidence
        self.roc_labels = []   # 1 = judge correct, 0 = judge incorrect
        self.dedup_stats = {"unique_groups": 0, "total_in_groups": 0}

    def record_roc_observation(self, label, active_predictions):
        """
        Store ROC-ready observations only for binary review outcomes.
        label=1 means the judge was considered correct by the human reviewer.
        label=0 means the judge was considered incorrect.
        """
        self.roc_labels.append(label)
        if active_predictions:
            self.roc_scores.append(max(p["conf"] for p in active_predictions))
        else:
            self.roc_scores.append(0.0)

    # ---------- RESIZE ----------
    def fit_to_screen(self, img, max_w=1280, max_h=720):
        h, w = img.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        return cv2.resize(img, (int(w*scale), int(h*scale))), scale

    def pad_to_height(self, img, target_h):
        h, w = img.shape[:2]
        if h >= target_h:
            return img
        pad = target_h - h
        return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    def hamming_distance_hex(self, h1, h2):
        """Calcula distancia de Hamming entre dos hashes hex."""
        if not h1 or not h2:
            return None
        if len(h1) != len(h2):
            return None
        return bin(int(h1, 16) ^ int(h2, 16)).count("1")

    # ---------- OVERLAY ----------
    def draw_overlay(self, img, gt_boxes, all_predictions, threshold):
        overlay = img.copy()
        h, w = img.shape[:2]
        active_preds = [p for p in all_predictions if p['conf'] >= threshold]

        for box in gt_boxes:
            cv2.rectangle(overlay, tuple(box[:2]), tuple(box[2:]), (0,255,0), 3)

        for pred in active_preds:
            x1,y1,x2,y2 = pred['box']
            cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(overlay,f"{pred['conf']:.2f}",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

        panel_h = int(h*0.18)
        panel = np.zeros((panel_h,w,3),dtype=np.uint8)
        panel[:] = (40,40,40)

        texts = [
            f"THRESHOLD: {threshold:.2f}  [N/↑ subir | B/↓ bajar | ESPACIO aplicar | ESC salir]",
            f"HUMANO: {len(gt_boxes)}   JUEZ: {len(active_preds)}   TOTAL: {len(all_predictions)}"
        ]

        for i,t in enumerate(texts):
            cv2.putText(panel,t,(10,30+i*25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),1)

        return np.vstack([panel,overlay])

    def draw_gt_boxes(self, img, boxes, color=(0, 255, 0), thickness=2):
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return img

    def draw_pred_boxes(self, img, preds, color=(0, 0, 255), thickness=2):
        for pred in preds:
            x1, y1, x2, y2 = pred['box']
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(img, f"{pred['conf']:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img

    # ---------- PREDICCIONES ----------
    def get_all_predictions(self, img_path, w, h):
        results = self.model.predict(str(img_path), conf=0.10, verbose=False, classes=[0])
        preds=[]
        if results[0].boxes:
            for i,box in enumerate(results[0].boxes.xywh):
                conf=results[0].boxes.conf[i].item()
                xc,yc,bw,bh=box.cpu().tolist()
                box_abs=utils.xywh_norm_to_xyxy_abs([xc/w,yc/h,bw/w,bh/h],w,h)
                preds.append({'box':box_abs,'conf':conf})
        return preds

    # ---------- LOOP ----------
    def inspect_by_label_issue(self, label_values, limit=10):
        from ultralytics import YOLO
        self.model = YOLO(config.MODEL_JUDGE_PATH)

        conn = sqlite3.connect(config.DB_PATH)
        cur = conn.cursor()

        placeholders = ",".join(["?"] * len(label_values))
        rows = cur.execute(f"""
            SELECT id, relative_path, width, height
            FROM images
            WHERE label_issue IN ({placeholders})
            ORDER BY RANDOM()
            LIMIT ?
        """, (*label_values, limit)).fetchall()

        for img_id, rel_path, w, h in rows:
            img_path = config.DATASET_ROOT / rel_path
            img_raw = cv2.imread(str(img_path))
            img, scale = self.fit_to_screen(img_raw)

            anns = cur.execute(
                "SELECT x_center,y_center,width,height FROM annotations WHERE image_id=?",
                (img_id,)
            ).fetchall()
            gt_boxes=[utils.xywh_norm_to_xyxy_abs(list(a),w,h) for a in anns]
            gt_boxes=[[int(x1*scale),int(y1*scale),int(x2*scale),int(y2*scale)] for x1,y1,x2,y2 in gt_boxes]

            all_preds=self.get_all_predictions(img_path,w,h)
            for p in all_preds:
                p['box']=[int(v*scale) for v in p['box']]

            self.current_threshold=config.CONF_THRESHOLD
            win=f"Inspector-{img_id}"

            while True:
                disp=self.draw_overlay(img,gt_boxes,all_preds,self.current_threshold)
                cv2.imshow(win,disp)
                key=cv2.waitKey(50)&0xFF

                if key in (82,ord('n'),ord('N')):
                    self.current_threshold=min(0.95,self.current_threshold+0.05)
                elif key in (84,ord('b'),ord('B')):
                    self.current_threshold=max(0.10,self.current_threshold-0.05)
                elif key==32:
                    cv2.destroyWindow(win); break
                elif key==27:
                    cv2.destroyAllWindows(); conn.close(); return

            active=[p for p in all_preds if p['conf']>=self.current_threshold]

            print("\n[1] TRUE ERROR  [2] FALSE ALARM  [3] AMBIGUOUS  [s] SKIP  [q] QUIT")
            final=self.draw_overlay(img,gt_boxes,all_preds,self.current_threshold)
            cv2.imshow("Diagnóstico",final)
            key=cv2.waitKey(0)&0xFF
            cv2.destroyAllWindows()

            if key==ord('1'):
                self.findings["true_error"]+=1
                self.record_roc_observation(1, active)
            elif key==ord('2'):
                self.findings["false_alarm"]+=1
                self.record_roc_observation(0, active)
            elif key==ord('3'):
                self.findings["ambiguous"]+=1
            elif key==ord('s'):
                continue
            elif key==ord('q'):
                break

        conn.close()
        self.show_report()
        self.plot_roc()

    def inspect_bad_labels(self, limit=10):
        self.inspect_by_label_issue(["bad_label"], limit)

    def inspect_poor_alignment(self, limit=10):
        self.inspect_by_label_issue(["poor_alignment"], limit)

    def inspect_weak(self, limit=10):
        # "weak" may be labeled as "weak" or "ok" depending on the DB
        self.inspect_by_label_issue(["weak", "ok"], limit)

    # ---------- DEDUP (PHASH) ----------
    def inspect_dedup(self, limit=10):
        from ultralytics import YOLO
        if self.model is None:
            self.model = YOLO(config.MODEL_JUDGE_PATH)

        conn = sqlite3.connect(config.DB_PATH)
        cur = conn.cursor()

        # Take one representative per pHash group with duplicates
        rows = cur.execute("""
            SELECT MIN(id) as id, relative_path, width, height, phash
            FROM images
            WHERE phash IS NOT NULL AND phash != ''
            GROUP BY phash
            HAVING COUNT(*) > 1
            ORDER BY RANDOM()
            LIMIT ?
        """, (limit,)).fetchall()

        self.dedup_stats["unique_groups"] = len(rows)

        if not rows:
            print("[INFO] No hay grupos con duplicados por pHash.")
            conn.close()
            return

        # Count total images in those groups (for reporting)
        phashes = [r[4] for r in rows]
        placeholders = ",".join(["?"] * len(phashes))
        total_in_groups = cur.execute(
            f"SELECT COUNT(*) FROM images WHERE phash IN ({placeholders})",
            phashes
        ).fetchone()[0]
        self.dedup_stats["total_in_groups"] = total_in_groups

        for img_id, rel_path, w, h, phash in rows:
            # Fetch at least 2 images with the same pHash
            pair = cur.execute("""
                SELECT id, relative_path, width, height
                FROM images
                WHERE phash = ?
                ORDER BY id
                LIMIT 2
            """, (phash,)).fetchall()

            if len(pair) < 2:
                continue

            (id_a, rel_a, w_a, h_a), (id_b, rel_b, w_b, h_b) = pair
            img_a = cv2.imread(str(config.DATASET_ROOT / rel_a))
            img_b = cv2.imread(str(config.DATASET_ROOT / rel_b))

            if img_a is None or img_b is None:
                print(f"[WARN] No se pudo leer par: {rel_a} / {rel_b}")
                continue

            img_a, scale_a = self.fit_to_screen(img_a, max_w=900, max_h=600)
            img_b, scale_b = self.fit_to_screen(img_b, max_w=900, max_h=600)

            # Judge predictions with confidence
            preds_a = self.get_all_predictions(config.DATASET_ROOT / rel_a, w_a, h_a)
            preds_b = self.get_all_predictions(config.DATASET_ROOT / rel_b, w_b, h_b)

            for p in preds_a:
                p['box'] = [int(v * scale_a) for v in p['box']]
            for p in preds_b:
                p['box'] = [int(v * scale_b) for v in p['box']]

            img_a = self.draw_pred_boxes(img_a, preds_a)
            img_b = self.draw_pred_boxes(img_b, preds_b)

            # Match height for the horizontal subplot
            target_h = max(img_a.shape[0], img_b.shape[0])
            img_a = self.pad_to_height(img_a, target_h)
            img_b = self.pad_to_height(img_b, target_h)
            pair_img = np.hstack([img_a, img_b])

            win = f"Dedup-pHash {phash[:8]} (IDs {id_a}/{id_b})"

            # Simple overlay with info
            panel_h = int(pair_img.shape[0] * 0.18)
            panel = np.zeros((panel_h, pair_img.shape[1], 3), dtype=np.uint8)
            panel[:] = (40, 40, 40)
            ham = self.hamming_distance_hex(phash, phash)
            texts = [
                f"pHash: {phash}",
                f"Hamming: {ham if ham is not None else 'N/A'}",
                f"A: {id_a} {rel_a}",
                f"B: {id_b} {rel_b}",
                "ESPACIO siguiente | ESC salir"
            ]
            for i, t in enumerate(texts):
                cv2.putText(panel, t, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            disp = np.vstack([panel, pair_img])

            while True:
                cv2.imshow(win, disp)
                key = cv2.waitKey(50) & 0xFF
                if key == 32:
                    cv2.destroyWindow(win)
                    break
                if key == 27:
                    cv2.destroyAllWindows()
                    conn.close()
                    return

        conn.close()
        self.show_dedup_report()

    # ---------- ROC ----------
    def plot_roc(self):
        if not self.roc_labels or not self.roc_scores:
            print("\n[INFO] ROC skipped: no binary review decisions were recorded.")
            return

        unique_labels = set(self.roc_labels)
        if len(unique_labels) < 2:
            print(
                "\n[INFO] ROC skipped: both binary classes are required "
                f"but only {sorted(unique_labels)} were observed."
            )
            return

        fpr,tpr,_=roc_curve(self.roc_labels,self.roc_scores, pos_label=1)
        roc_auc=auc(fpr,tpr)
        plt.figure()
        plt.plot(fpr,tpr,label=f"AUC={roc_auc:.2f}")
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Humano vs Juez")
        plt.legend()
        plt.show()

    # ---------- REPORTE ----------
    def show_report(self):
        print("\nREPORTE FINAL")
        for k,v in self.findings.items():
            print(f"{k}: {v}")

    def show_dedup_report(self):
        print("\nREPORTE DEDUP (PHASH)")
        print(f"Grupos únicos con duplicados inspeccionados: {self.dedup_stats['unique_groups']}")
        print(f"Total imágenes en esos grupos: {self.dedup_stats['total_in_groups']}")

def main():
    inspector=InteractiveInspector()
    mode = input("Modo: [1] bad_labels  [2] dedup_phash  [3] poor_alignment  [4] weak  (default=1): ").strip()
    limit = input("¿Cuántas imágenes? (default=10): ").strip()
    limit_n = int(limit) if limit else 10
    if mode == "2":
        inspector.inspect_dedup(limit_n)
    elif mode == "3":
        inspector.inspect_poor_alignment(limit_n)
    elif mode == "4":
        inspector.inspect_weak(limit_n)
    else:
        inspector.inspect_bad_labels(limit_n)

if __name__=="__main__":
    main()
