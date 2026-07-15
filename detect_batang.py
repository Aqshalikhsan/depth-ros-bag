#!/usr/bin/env python3
"""Deteksi batang sawit (YOLO) + estimasi jarak monokular via pinhole lebar-bbox.

Alur:
  1. Jalankan YOLO `batang-best.pt` pada frame-frame dari ROS2 bag.
  2. Untuk tiap deteksi, catat lebar bbox (piksel) -> proxy diameter batang.
  3. Kalibrasi konstanta K = Z_gt * w_px dari scene diam (jarak GT diketahui).
  4. Estimasi jarak frame lain: Z = K / w_px.

Contoh:
  # 1) lihat deteksi + lebar bbox dulu (belum kalibrasi)
  python3 detect_batang.py <bag> -o out/ --step 15
  # 2) setelah tahu frame diem-nya, kalibrasi:
  python3 detect_batang.py <bag> -o out/ --step 15 --calib-frame 2400 --calib-dist 3.0

Catatan: dipilih bbox dengan pusat paling dekat ke tengah frame (paling minim
distorsi lensa wide + paling mungkin objek target), conf tertinggi sebagai tie-break.
"""
import argparse
import csv
import os

import cv2
import numpy as np
from ultralytics import YOLO

from extract_frames import find_db3, iter_images


def pick_main_det(boxes, img_w, img_h):
    """Pilih 1 bbox utama: paling dekat pusat frame, bobot conf."""
    if len(boxes) == 0:
        return None
    cx0, cy0 = img_w / 2, img_h / 2
    best, best_score = None, -1e9
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        conf = float(b.conf[0])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist_center = ((cx - cx0) ** 2 + (cy - cy0) ** 2) ** 0.5
        score = conf - 0.001 * dist_center  # dekat pusat + conf tinggi
        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2, conf)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('-o', '--out', default='out')
    ap.add_argument('--model', default='models/batang-best.pt')
    ap.add_argument('--topic', default='/camera_front')
    ap.add_argument('--step', type=int, default=15)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--save-vis', action='store_true', help='simpan frame teranotasi')
    ap.add_argument('--calib-frame', type=int, default=None)
    ap.add_argument('--calib-dist', type=float, default=None)
    ap.add_argument('--K', type=float, default=None, help='pakai K langsung (m*px) alih2 kalibrasi')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = YOLO(args.model)
    print('classes:', model.names)

    db3 = find_db3(args.bag)
    rows = []          # (idx, ts, w_px, h_px, conf, cx, cy)
    frames_cache = {}  # idx -> (img, det)  hanya untuk yang mau divisual/kalibrasi

    for idx, ts, img in iter_images(db3, args.topic):
        if idx % args.step:
            continue
        h, w = img.shape[:2]
        res = model.predict(img, conf=args.conf, verbose=False)[0]
        det = pick_main_det(res.boxes, w, h)
        if det is None:
            rows.append((idx, ts, None, None, None, None, None))
            continue
        x1, y1, x2, y2, conf = det
        wpx, hpx = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rows.append((idx, ts, wpx, hpx, conf, cx, cy))
        if args.save_vis or idx == args.calib_frame:
            frames_cache[idx] = (img.copy(), det)

    # --- kalibrasi K ---
    K = args.K
    if K is None and args.calib_frame is not None and args.calib_dist is not None:
        match = [r for r in rows if r[0] == args.calib_frame and r[2] is not None]
        if match:
            wpx = match[0][2]
            K = args.calib_dist * wpx
            print(f'[KALIBRASI] frame {args.calib_frame}: w={wpx:.1f}px @ {args.calib_dist}m '
                  f'-> K = {K:.1f} (m*px)')
        else:
            print(f'[KALIBRASI] tidak ada deteksi di frame {args.calib_frame}!')

    # --- tulis CSV + estimasi ---
    csv_path = os.path.join(args.out, 'detections.csv')
    with open(csv_path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['frame', 'timestamp_ns', 'w_px', 'h_px', 'conf', 'cx', 'cy', 'Z_est_m'])
        for (idx, ts, wpx, hpx, conf, cx, cy) in rows:
            z = (K / wpx) if (K and wpx) else ''
            wr.writerow([idx, ts,
                         f'{wpx:.1f}' if wpx else '',
                         f'{hpx:.1f}' if hpx else '',
                         f'{conf:.3f}' if conf else '',
                         f'{cx:.1f}' if cx else '',
                         f'{cy:.1f}' if cy else '',
                         f'{z:.2f}' if z != '' else ''])
    n_det = sum(1 for r in rows if r[2] is not None)
    print(f'CSV: {csv_path}  ({n_det}/{len(rows)} frame terdeteksi)')

    # --- visual ---
    if args.save_vis:
        vdir = os.path.join(args.out, 'vis')
        os.makedirs(vdir, exist_ok=True)
        for idx, (img, det) in frames_cache.items():
            x1, y1, x2, y2, conf = det
            wpx = x2 - x1
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f'w={wpx:.0f}px conf={conf:.2f}'
            if K:
                label += f'  Z={K/wpx:.2f}m'
            cv2.putText(img, label, (int(x1), max(15, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(vdir, f'vis_{idx:05d}.png'), img)
        print(f'Visual: {vdir}/')


if __name__ == '__main__':
    main()
