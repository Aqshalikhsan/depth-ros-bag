#!/usr/bin/env python3
"""Ekstrak frame gambar dari ROS2 bag (.db3) topik sensor_msgs/Image (bgr8).

Tanpa dependency ROS: baca SQLite langsung + parser CDR minimal untuk Image.
Contoh:
    python3 extract_frames.py <bag_dir_or_db3> -o frames --step 30
"""
import argparse
import glob
import os
import sqlite3
import struct

import numpy as np
from PIL import Image


def _align(off, a):
    r = off % a
    return off + (a - r) if r else off


def parse_image(b):
    """Deserialize CDR sensor_msgs/msg/Image -> (stamp_ns, np.ndarray HxWx3 BGR)."""
    off = 4  # skip 4-byte encapsulation header
    off = _align(off, 4)
    sec = struct.unpack_from('<i', b, off)[0]; off += 4
    nsec = struct.unpack_from('<I', b, off)[0]; off += 4
    off = _align(off, 4); slen = struct.unpack_from('<I', b, off)[0]; off += 4
    off += slen  # frame_id
    off = _align(off, 4)
    height = struct.unpack_from('<I', b, off)[0]; off += 4
    width = struct.unpack_from('<I', b, off)[0]; off += 4
    off = _align(off, 4); elen = struct.unpack_from('<I', b, off)[0]; off += 4
    enc = b[off:off + elen].split(b'\x00')[0].decode('latin1'); off += elen
    off += 1  # is_bigendian
    off = _align(off, 4); step = struct.unpack_from('<I', b, off)[0]; off += 4
    off = _align(off, 4); dlen = struct.unpack_from('<I', b, off)[0]; off += 4
    data = np.frombuffer(b, dtype=np.uint8, count=dlen, offset=off)
    img = data.reshape(height, width, 3)
    return sec * 1_000_000_000 + nsec, enc, img  # img is BGR (bgr8)


def find_db3(path):
    if path.endswith('.db3'):
        return path
    hits = sorted(glob.glob(os.path.join(path, '*.db3')))
    if not hits:
        raise SystemExit(f'Tidak ada .db3 di {path}')
    return hits[0]


def iter_images(db3, topic='/camera_front'):
    con = sqlite3.connect(db3)
    cur = con.cursor()
    cur.execute("SELECT id FROM topics WHERE name=?", (topic,))
    row = cur.fetchone()
    if not row:
        raise SystemExit(f'Topik {topic} tidak ada di {db3}')
    tid = row[0]
    for i, (ts, blob) in enumerate(
            cur.execute("SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,))):
        stamp_ns, enc, img = parse_image(blob)
        yield i, ts, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('-o', '--out', default='frames')
    ap.add_argument('--topic', default='/camera_front')
    ap.add_argument('--step', type=int, default=30, help='ambil 1 frame tiap N (default 30 ~ 1fps)')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=-1)
    args = ap.parse_args()

    db3 = find_db3(args.bag)
    os.makedirs(args.out, exist_ok=True)
    n = 0
    for i, ts, img in iter_images(db3, args.topic):
        if i < args.start:
            continue
        if args.end >= 0 and i > args.end:
            break
        if i % args.step:
            continue
        rgb = img[:, :, ::-1]  # BGR->RGB untuk disimpan
        Image.fromarray(rgb).save(os.path.join(args.out, f'frame_{i:05d}.png'))
        n += 1
    print(f'Selesai: {n} frame -> {args.out}/  (dari {db3})')


if __name__ == '__main__':
    main()
