#!/usr/bin/env python3
"""Simulasi vision-only motion/looming: jarak relatif + meter(asumsi) + TTC.

Memakai MotionDistanceEstimator (melacak 1 batang). Tidak menimpa sim_realtime.py.

Contoh:
  python3 sim_motion.py <bag_atau_.db3> -o out/motion --every 5 --K-nom 193.5
"""
import argparse
import csv
import os
import shutil
import subprocess

import cv2
from ultralytics import YOLO

from extract_frames import find_db3, iter_images
from motion_estimator import MotionDistanceEstimator


def draw(img, r):
    h, w = img.shape[:2]
    det = r['detection']
    if det:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(img, (0, h - 62), (300, h), (0, 0, 0), -1)
    if r['z_rel'] is not None:
        cv2.putText(img, f"rel: {r['z_rel']:.2f}x  (~{r['z_m']:.2f} m*)",
                    (8, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        ttc = r['ttc_s']
        if ttc is not None:
            col = (0, 0, 255) if ttc < 2.0 else (0, 220, 255)
            cv2.putText(img, f"TTC: {ttc:.1f} s", (8, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        else:
            cv2.putText(img, "TTC: -- (tak mendekat)", (8, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    else:
        cv2.putText(img, "cari batang...", (8, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(img, "* meter = asumsi skala", (w - 210, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('-o', '--out', default='out/motion')
    ap.add_argument('--model', default='models/batang-best.pt')
    ap.add_argument('--topic', default='/camera_front')
    ap.add_argument('--every', type=int, default=5)
    ap.add_argument('--K-nom', type=float, default=193.5)
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--fps', type=float, default=15.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    est = MotionDistanceEstimator(YOLO(args.model), K_nom=args.K_nom, conf=args.conf)
    db3 = find_db3(args.bag)

    writer = None
    vpath = os.path.join(args.out, 'motion.mp4')
    f = open(os.path.join(args.out, 'motion.csv'), 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(['frame', 'timestamp_ns', 'w_px', 'z_rel', 'z_m_assumed', 'ttc_s', 'closing'])

    n = 0
    for idx, ts, img in iter_images(db3, args.topic):
        if idx % args.every:
            continue
        r = est.update(img, ts / 1e9)
        wr.writerow([idx, ts,
                     f"{r['w_px']:.1f}" if r['w_px'] else '',
                     f"{r['z_rel']:.3f}" if r['z_rel'] is not None else '',
                     f"{r['z_m']:.2f}" if r['z_m'] is not None else '',
                     f"{r['ttc_s']:.2f}" if r['ttc_s'] is not None else '',
                     r['closing'] if r['closing'] is not None else ''])
        vis = draw(img.copy(), r)
        if writer is None:
            hh, ww = vis.shape[:2]
            writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (ww, hh))
        writer.write(vis)
        n += 1

    if writer:
        writer.release()
    f.close()
    if writer and shutil.which('ffmpeg'):
        h264 = vpath.replace('.mp4', '_h264.mp4')
        try:
            subprocess.run(['ffmpeg', '-y', '-i', vpath, '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', h264],
                           check=True, capture_output=True)
            print(f'H.264: {h264}')
        except subprocess.CalledProcessError:
            pass
    print(f'Selesai: {n} frame -> {args.out}/')


if __name__ == '__main__':
    main()
