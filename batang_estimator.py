#!/usr/bin/env python3
"""Estimator jarak drone->batang sawit, monokular, real-time (vision-only).

Metode: pinhole known-width -> Z = K / lebar_bbox_px, dengan K dikalibrasi
sekali dari scene diem ber-GT. Ditambah pemilihan target + filter temporal
supaya output stabil saat drone terbang.

Dipakai bersama oleh:
  - sim_realtime.py   (validasi offline dgn bag)
  - ros2_batang_node.py (runtime di drone)

Kalibrasi (default K) diturunkan dari scene hover ~3m bag 11_46_30:
  w_hover ~= 64.5 px @ 3.0 m  ->  K ~= 193.5
"""
from collections import deque

import numpy as np


class BatangDistanceEstimator:
    def __init__(self, model, K=193.5, conf=0.35,
                 select='largest', img_center=(320, 180),
                 filter_win=5, min_valid=2, miss_reset=8):
        """
        model      : ultralytics YOLO (sudah dimuat) -- diinjeksi agar mudah dites
        K          : konstanta kalibrasi (m*px). Z = K / w_px
        conf       : ambang confidence deteksi
        select     : 'largest' (batang terdekat=bbox terlebar, utk hindar tabrakan)
                     | 'center' (batang paling dekat pusat frame)
        filter_win : jendela median temporal (jumlah DETEKSI, bukan frame)
        min_valid  : min. deteksi dalam jendela sebelum keluarkan angka
        miss_reset : reset histori jika sekian frame beruntun tak terdeteksi

        Filter = median dari `filter_win` deteksi terakhir. Median menolak spike
        1-frame TAPI tetap pulih & mengikuti perubahan nyata (mis. drone mendekat
        cepat) -- tak ada penolakan-lonjakan keras yang bisa membekukan output.
        """
        self.model = model
        self.K = K
        self.conf = conf
        self.select = select
        self.cx0, self.cy0 = img_center
        self.hist = deque(maxlen=filter_win)   # raw Z deteksi terakhir
        self.min_valid = min_valid
        self.miss_reset = miss_reset
        self._miss = 0

    def _pick(self, boxes, w, h):
        if len(boxes) == 0:
            return None
        best, best_key = None, None
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            wpx = x2 - x1
            if self.select == 'center':
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                d = ((cx - self.cx0) ** 2 + (cy - self.cy0) ** 2) ** 0.5
                key = conf - 0.001 * d
            else:  # 'largest' = terdekat
                key = wpx
            if best_key is None or key > best_key:
                best_key, best = key, (x1, y1, x2, y2, conf, wpx)
        return best

    def update(self, img_bgr):
        """Proses 1 frame. Return dict hasil real-time."""
        h, w = img_bgr.shape[:2]
        res = self.model.predict(img_bgr, conf=self.conf, verbose=False)[0]
        det = self._pick(res.boxes, w, h)

        raw_z = None
        det_out = None
        if det is not None:
            x1, y1, x2, y2, conf, wpx = det
            raw_z = self.K / wpx if wpx > 0 else None
            det_out = dict(bbox=(x1, y1, x2, y2), conf=conf, w_px=wpx)

        # --- filter temporal: median dari deteksi terakhir ---
        if raw_z is not None:
            self._miss = 0
            self.hist.append(raw_z)
        else:
            self._miss += 1
            if self._miss >= self.miss_reset:
                self.hist.clear()  # target hilang lama -> jangan lapor jarak basi

        z_filt = None
        if len(self.hist) >= self.min_valid:
            z_filt = float(np.median(self.hist))

        return dict(
            raw_z=raw_z,          # estimasi mentah frame ini (None jika tak terdeteksi)
            z=z_filt,             # jarak ter-filter (yang dipakai/di-publish)
            detection=det_out,    # bbox+conf+w_px atau None
            n_hist=len(self.hist),
        )
