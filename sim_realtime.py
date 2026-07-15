#!/usr/bin/env python3
"""Simulasi real-time: putar ulang bag lewat BatangDistanceEstimator seolah live.

Membuktikan perilaku runtime (yang nanti dipakai node ROS2 di drone) tanpa ROS.
Keluaran: video .mp4 teranotasi + CSV jarak ter-filter per frame.

Contoh:
  python3 sim_realtime.py <bag> -o out/sim --every 2 --K 193.5
"""
import argparse
import csv
import os
import shutil
import subprocess

import cv2
from ultralytics import YOLO

from batang_estimator import BatangDistanceEstimator
from extract_frames import find_db3, iter_images


def draw(img, r):
    h, w = img.shape[:2]
    det = r['detection']
    if det:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # panel teks
    z = r['z']
    txt = f"JARAK: {z:0.2f} m" if z is not None else "JARAK: -- (cari batang)"
    color = (0, 255, 0) if z is not None else (0, 200, 255)
    if z is not None and z < 1.5:
        color = (0, 0, 255)  # merah = terlalu dekat
    cv2.rectangle(img, (0, h - 34), (270, h), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if r['raw_z'] is not None:
        cv2.putText(img, f"raw {r['raw_z']:.2f}", (w - 120, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('-o', '--out', default='out/sim')
    ap.add_argument('--model', default='models/batang-best.pt')
    ap.add_argument('--topic', default='/camera_front')
    ap.add_argument('--every', type=int, default=2, help='proses tiap N frame (real-time throttle)')
    ap.add_argument('--K', type=float, default=193.5)
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--select', default='largest', choices=['largest', 'center'])
    ap.add_argument('--fps', type=float, default=15.0, help='fps video keluaran')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    est = BatangDistanceEstimator(YOLO(args.model), K=args.K, conf=args.conf, select=args.select)
    db3 = find_db3(args.bag)

    writer = None
    vpath = os.path.join(args.out, 'realtime.mp4')
    csv_path = os.path.join(args.out, 'realtime.csv')
    f = open(csv_path, 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(['frame', 'timestamp_ns', 'raw_z', 'z_filt', 'w_px', 'conf'])

    n = 0
    for idx, ts, img in iter_images(db3, args.topic):
        if idx % args.every:
            continue
        r = est.update(img)
        det = r['detection']
        wr.writerow([idx, ts,
                     f"{r['raw_z']:.3f}" if r['raw_z'] is not None else '',
                     f"{r['z']:.3f}" if r['z'] is not None else '',
                     f"{det['w_px']:.1f}" if det else '',
                     f"{det['conf']:.3f}" if det else ''])
        vis = draw(img.copy(), r)
        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
        writer.write(vis)
        n += 1

    if writer:
        writer.release()
    f.close()

    # transcode ke H.264 (yuv420p) supaya bisa dibuka di player laptop & browser
    if writer and shutil.which('ffmpeg'):
        h264 = vpath.replace('.mp4', '_h264.mp4')
        try:
            subprocess.run(['ffmpeg', '-y', '-i', vpath, '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', h264],
                           check=True, capture_output=True)
            print(f'H.264 (kompatibel): {h264}')
        except subprocess.CalledProcessError:
            print('ffmpeg transcode gagal; pakai mp4v mentah.')

    print(f'Selesai: {n} frame diproses -> {vpath}  &  {csv_path}')


if __name__ == '__main__':
    main()
