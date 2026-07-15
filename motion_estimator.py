#!/usr/bin/env python3
"""Estimator jarak vision-only berbasis pelacakan pertumbuhan batang (looming).

Bedanya dari batang_estimator.py (metode K/w statis):
  - Melacak SATU batang antar-frame (asosiasi IoU) -> konsisten saat terbang.
  - Output UTAMA = jarak RELATIF (dimensionless) dari rasio lebar bbox -> EKSAK,
    tak butuh skala/GT.
  - Output METER = K_nom/w, berlabel "asumsi skala" (perubahan relatif akurat,
    nilai absolut bergantung K_nom).
  - Time-to-Contact (TTC, detik) = w / (dw/dt) -> MURNI vision-only, tanpa skala,
    berguna untuk hindar-tabrakan.

Tidak menimpa file lain. Dipakai oleh sim_motion.py (dan bisa oleh node ROS).
"""
from collections import deque

import numpy as np


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


class MotionDistanceEstimator:
    def __init__(self, model, K_nom=193.5, conf=0.35,
                 win=7, miss_reset=8, ttc_min_slope=0.5):
        """
        model        : ultralytics YOLO (sudah dimuat)
        K_nom         : skala nominal utk TAMPILAN meter (Z_m = K_nom/w). ASUMSI.
        conf          : ambang confidence
        win           : jendela (jumlah deteksi) utk regresi dw/dt & filter
        miss_reset    : reset track jika sekian frame beruntun tak terdeteksi
        ttc_min_slope : dw/dt minimum (px/s) agar TTC dihitung (hindari bagi ~0)
        """
        self.model = model
        self.K_nom = K_nom
        self.conf = conf
        self.miss_reset = miss_reset
        self.ttc_min_slope = ttc_min_slope
        self.hist = deque(maxlen=win)   # (t_sec, w_px)
        self.track_bbox = None          # bbox target terakhir (untuk asosiasi)
        self.w_ref = None               # lebar saat track dimulai (acuan relatif)
        self._miss = 0

    def _associate(self, boxes):
        """Pilih deteksi yang paling cocok dgn track sebelumnya; kalau belum ada
        track, mulai dari bbox terbesar (batang terdekat)."""
        cand = []
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cand.append((x1, y1, x2, y2, float(b.conf[0])))
        if not cand:
            return None
        if self.track_bbox is None:
            return max(cand, key=lambda c: (c[2] - c[0]))  # terbesar
        # cocokkan via IoU tertinggi dgn track lama
        best = max(cand, key=lambda c: _iou(self.track_bbox, c[:4]))
        if _iou(self.track_bbox, best[:4]) < 0.1:
            return max(cand, key=lambda c: (c[2] - c[0]))   # track hilang -> reset ke terbesar
        return best

    def update(self, img_bgr, t_sec):
        h, w = img_bgr.shape[:2]
        res = self.model.predict(img_bgr, conf=self.conf, verbose=False)[0]
        det = self._associate(res.boxes)

        if det is None:
            self._miss += 1
            if self._miss >= self.miss_reset:
                self.hist.clear()
                self.track_bbox = None
                self.w_ref = None
            return dict(detection=None, z_rel=None, z_m=None, ttc_s=None,
                        w_px=None, closing=None)

        self._miss = 0
        x1, y1, x2, y2, conf = det
        wpx = x2 - x1
        self.track_bbox = (x1, y1, x2, y2)
        if self.w_ref is None:
            self.w_ref = wpx
        self.hist.append((t_sec, wpx))

        # jarak relatif (dimensionless): 1.0 di awal track, mengecil saat mendekat
        z_rel = self.w_ref / wpx if wpx > 0 else None
        # meter (ASUMSI skala K_nom)
        z_m = self.K_nom / wpx if wpx > 0 else None

        # dw/dt via regresi linear pd jendela -> TTC
        ttc_s, closing = None, None
        if len(self.hist) >= 3:
            ts = np.array([p[0] for p in self.hist])
            ws = np.array([p[1] for p in self.hist])
            ts = ts - ts[0]
            slope = np.polyfit(ts, ws, 1)[0]  # px/s
            closing = slope > 0
            if slope > self.ttc_min_slope:
                ttc_s = wpx / slope  # detik sampai "kontak" (w -> besar)

        return dict(detection=dict(bbox=(x1, y1, x2, y2), conf=conf, w_px=wpx),
                    z_rel=z_rel, z_m=z_m, ttc_s=ttc_s, w_px=wpx, closing=closing)
