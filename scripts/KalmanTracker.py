import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import scipy.linalg


class KalmanFilterXYAH:
    """Filtro de Kalman con estado [x, y, a, h, vx, vy, va, vh]
    x, y = centro del bbox
    a = aspect ratio (w / h)
    h = altura del bbox
    """

    def __init__(self):
        ndim, dt = 4, 1.0

        # Matriz de transición (CV: velocidad constante)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Matriz de observación (observamos x, y, a, h)
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Pesos de incertidumbre (relativos a la escala del objeto)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def _safe_h(self, h):
        # Evita valores degenerados de altura
        return float(max(h, 1e-2))

    def initiate(self, measurement):
        """Inicializa el track a partir de una medición [x, y, a, h]."""
        mx, my, ma, mh = measurement
        mh = self._safe_h(mh)

        mean_pos = np.array([mx, my, ma, mh], dtype=float)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * mh,
            2 * self._std_weight_position * mh,
            1e-2,
            2 * self._std_weight_position * mh,
            10 * self._std_weight_velocity * mh,
            10 * self._std_weight_velocity * mh,
            1e-5,
            10 * self._std_weight_velocity * mh,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Predice el siguiente estado."""
        h = self._safe_h(mean[3])

        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        """Proyecta el estado al espacio de medición."""
        h = self._safe_h(mean[3])
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        innovation_cov = np.diag(np.square(std))

        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Actualiza el estado con una nueva medición [x, y, a, h]."""
        projected_mean, projected_cov = self.project(mean, covariance)

        # Robustez numérica
        projected_cov = projected_cov.copy()
        projected_cov[np.diag_indices_from(projected_cov)] += 1e-6

        # K = PHT (HPHT + R)^{-1}  — resolvemos con Cholesky por estabilidad
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        K = scipy.linalg.cho_solve((chol_factor, lower), (covariance @ self._update_mat.T).T, check_finite=False).T

        innovation = measurement - projected_mean
        new_mean = mean + K @ innovation
        new_covariance = covariance - K @ projected_cov @ K.T
        return new_mean, new_covariance


class KalmanTracker:
    """Tracker basado en Filtro de Kalman (estado XYAH).

    Entradas a update(): lista de detecciones en formato [x, y, w, h, score, class].
    Salida de update(): lista de tracks activos en formato [x, y, w, h, track_id, class, score].
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        min_conf_new_track: float = 0.10,
        min_conf_keep_track: float = 0.05,
    ):
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.min_conf_new_track = float(min_conf_new_track)
        self.min_conf_keep_track = float(min_conf_keep_track)

        self.kf = KalmanFilterXYAH()
        self.next_id = 1
        self.tracks = defaultdict(dict)
        self.frame_count = 0

    # ----------------------------- API pública -----------------------------
    def reset(self):
        """Reinicia el estado del tracker (útil entre videos)."""
        self.next_id = 1
        self.tracks.clear()
        self.frame_count = 0

    def update(self, detections):
        """Actualiza el tracker con nuevas detecciones.

        detections: List[List[x, y, w, h, score, class]]  (x, y son centro)
        return: List[List[x, y, w, h, track_id, class, score]]
        """
        self.frame_count += 1

        # 1) Predicción de todos los tracks existentes
        to_remove = []
        for track_id, track in list(self.tracks.items()):
            mean, cov = self.kf.predict(track['mean'], track['covariance'])
            track['mean'] = mean
            track['covariance'] = cov
            track['age'] += 1
            track['total_frames'] += 1
            track['predicted_bbox'] = self._convert_state_to_bbox(mean)
            if track['age'] > self.max_age:
                to_remove.append(track_id)
        for tid in to_remove:
            self.tracks.pop(tid, None)

        if detections is None:
            detections = []

        # Si no hay detecciones, devolvemos los tracks confirmados (ya predichos)
        if len(detections) == 0:
            return self._get_active_tracks()

        # 2) Asociación detecciones↔tracks mediante IoU + Hungarian
        track_ids = list(self.tracks.keys())
        if len(track_ids) > 0:
            track_boxes = [self.tracks[tid]['predicted_bbox'] for tid in track_ids]
            det_boxes = [det[:4] for det in detections]

            iou_matrix = self._iou_batch(track_boxes, det_boxes)
            cost_matrix = 1.0 - iou_matrix

            rows, cols = linear_sum_assignment(cost_matrix)
            matched_pairs = []
            matched_tids, matched_dids = set(), set()
            for r, c in zip(rows, cols):
                if cost_matrix[r, c] < (1.0 - self.iou_threshold):
                    matched_pairs.append((r, c))
                    matched_tids.add(r)
                    matched_dids.add(c)

            # 2.a) Actualizar tracks emparejados
            for r, c in matched_pairs:
                tid = track_ids[r]
                det = detections[c]
                meas = self._convert_bbox_to_measurement(det[:4])
                mean, cov = self.kf.update(self.tracks[tid]['mean'], self.tracks[tid]['covariance'], meas)
                t = self.tracks[tid]
                t['mean'] = mean
                t['covariance'] = cov
                t['age'] = 0
                t['hits'] += 1
                t['score'] = float(det[4])
                t['class'] = int(det[5]) if len(det) > 5 else t.get('class', 0)

            # 2.b) Detecciones no asignadas → posibles nuevos tracks
            unmatched_dets = [i for i in range(len(detections)) if i not in matched_dids]
        else:
            unmatched_dets = list(range(len(detections)))

        # 3) Iniciar nuevos tracks para detecciones no asignadas con score suficiente
        for i in unmatched_dets:
            det = detections[i]
            if det[4] < self.min_conf_new_track:
                continue
            meas = self._convert_bbox_to_measurement(det[:4])
            mean, cov = self.kf.initiate(meas)
            self.tracks[self.next_id] = {
                'mean': mean,
                'covariance': cov,
                'age': 0,
                'hits': 1,
                'total_frames': 1,
                'score': float(det[4]),
                'class': int(det[5]) if len(det) > 5 else 0,
                'predicted_bbox': det[:4],
            }
            self.next_id += 1

        # 4) Devolver tracks confirmados
        return self._get_active_tracks()

    # --------------------------- Utilidades internas ---------------------------
    def _get_active_tracks(self):
        active = []
        for tid, t in self.tracks.items():
            if t['hits'] >= self.min_hits and t.get('score', 0.0) >= self.min_conf_keep_track:
                x, y, w, h = self._convert_state_to_bbox(t['mean'])
                active.append([float(x), float(y), float(w), float(h), int(tid), int(t.get('class', 0)), float(t.get('score', 0.0))])
        return active

    @staticmethod
    def _convert_bbox_to_measurement(bbox):
        """[x, y, w, h] (centro) → [x, y, a, h] con a = w/h (protegido)."""
        x, y, w, h = map(float, bbox)
        h = max(h, 1e-2)
        a = w / h
        return np.array([x, y, a, h], dtype=float)

    @staticmethod
    def _convert_state_to_bbox(state):
        """[x, y, a, h, ...] → [x, y, w, h] (centro)."""
        x, y, a, h = map(float, state[:4])
        h = max(h, 1e-2)
        w = a * h
        return np.array([x, y, w, h], dtype=float)

    @staticmethod
    def _iou_batch(bboxes1, bboxes2):
        """Calcula IoU entre dos listas de bboxes [x, y, w, h] (centro).
        Devuelve matriz (N x M).
        """
        if len(bboxes1) == 0 or len(bboxes2) == 0:
            return np.zeros((len(bboxes1), len(bboxes2)), dtype=float)

        b1 = np.asarray(bboxes1, dtype=float)
        b2 = np.asarray(bboxes2, dtype=float)

        # Convertir a esquinas (x1, y1, x2, y2)
        b1_xyxy = np.stack([
            b1[:, 0] - b1[:, 2] / 2.0,
            b1[:, 1] - b1[:, 3] / 2.0,
            b1[:, 0] + b1[:, 2] / 2.0,
            b1[:, 1] + b1[:, 3] / 2.0,
        ], axis=1)
        b2_xyxy = np.stack([
            b2[:, 0] - b2[:, 2] / 2.0,
            b2[:, 1] - b2[:, 3] / 2.0,
            b2[:, 0] + b2[:, 2] / 2.0,
            b2[:, 1] + b2[:, 3] / 2.0,
        ], axis=1)

        area1 = np.maximum(0.0, b1_xyxy[:, 2] - b1_xyxy[:, 0]) * np.maximum(0.0, b1_xyxy[:, 3] - b1_xyxy[:, 1])
        area2 = np.maximum(0.0, b2_xyxy[:, 2] - b2_xyxy[:, 0]) * np.maximum(0.0, b2_xyxy[:, 3] - b2_xyxy[:, 1])

        lt = np.maximum(b1_xyxy[:, None, :2], b2_xyxy[None, :, :2])
        rb = np.minimum(b1_xyxy[:, None, 2:], b2_xyxy[None, :, 2:])
        wh = np.clip(rb - lt, a_min=0.0, a_max=None)
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2[None, :] - inter
        # Evitar división por cero
        union = np.maximum(union, 1e-9)
        return inter / union
