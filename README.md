# Estimasi Jarak Drone → Batang Sawit (Monokular, Vision-Only)

Estimasi jarak drone ke batang sawit **hanya dari kamera** (monokular RGB), tanpa
depth sensor / stereo / LiDAR. Dipakai untuk berjalan **real-time saat drone terbang**.

## Cara kerja (ringkas)

Model pinhole + ukuran objek diketahui:

```
Z (meter) = K / lebar_bbox_px          K = f · W  (dikalibrasi sekali)
```

- YOLO (`models/batang-best.pt`) mendeteksi batang → ambil **lebar** bounding box (px).
- `K` dikalibrasi sekali dari scene diam ber-ground-truth (hover ~3 m). **K ≈ 193.5**.
- Hasil di-**filter median** agar stabil saat terbang.
- **Bukan depth-net** — murni geometri, jadi ringan & real-time.

> Batasan: mengasumsikan diameter batang ~konstan & lensa rektilinear. Lensa FPV
> wide belum di-undistort → estimasi di tepi frame agak bias.

---

## 1. Setup (mesin analisis / laptop)

Tanpa venv (pakai user-site). Paket kunci:

```bash
sudo apt install -y python3-opencv ffmpeg sqlite3
pip3 install --user "numpy<2" ultralytics        # numpy<2 wajib: cocok dgn opencv apt
# torch + CUDA diasumsikan sudah ada
```

Cek:
```bash
python3 -c "import cv2, numpy, torch, ultralytics; print('OK', torch.cuda.is_available())"
```

> `numpy<2` penting: opencv apt (4.5.4) tidak kompatibel numpy 2.x.

---

## 2. Ekstrak frame dari rosbag (.db3)

Membaca ROS2 bag langsung via SQLite + parser CDR Image (tanpa perlu ROS terpasang).

```bash
python3 extract_frames.py <bag_dir_atau_.db3> -o frames/ --step 30
# --step 30 = ~1 frame/detik (kamera 30fps). --step 1 = semua frame.
```

---

## 3. Deteksi + estimasi jarak (batch/analisis)

```bash
# lihat deteksi & lebar bbox dulu (belum kalibrasi):
python3 detect_batang.py <bag> -o out/ --step 15 --save-vis

# kalibrasi K dari frame diem ber-GT (mis. frame 2400 = 3 m):
python3 detect_batang.py <bag> -o out/ --step 15 --calib-frame 2400 --calib-dist 3.0

# atau langsung pakai K yang sudah diketahui:
python3 detect_batang.py <bag> -o out/ --step 15 --K 193.5
```
Keluaran: `out/detections.csv` (kolom `Z_est_m`) + `out/vis/` (frame teranotasi).

---

## 4. Simulasi real-time (validasi offline + video)

Memutar ulang bag lewat estimator seolah live → **membuktikan perilaku runtime**
yang sama dengan node drone. Menghasilkan video + CSV jarak per frame.

```bash
python3 sim_realtime.py <bag> -o out/sim/ --every 5 --K 193.5 --select largest
```
Keluaran:
- `out/sim/realtime.mp4` (mp4v, mentah)
- `out/sim/realtime_h264.mp4` ← **buka ini** di VLC / browser (kompatibel)
- `out/sim/realtime.csv` (raw_z, z_filt, w_px, conf)

GIF preview (bisa dilihat inline di VSCode):
```bash
ffmpeg -y -i out/sim/realtime.mp4 -vf "select='between(n,105,185)',setpts=N/15/TB,scale=480:-1" -an out/sim/approach.gif
```

---

## 4b. Estimasi vision-only: jarak relatif + TTC (tanpa GT, tanpa skala)

Metode alternatif berbasis **motion/looming** — melacak SATU batang dan memakai
pertumbuhan lebar bbox. Berguna saat GT tidak diketahui / drone bergerak acak.

```bash
python3 sim_motion.py <bag_atau_.db3> -o out/motion/ --every 5 --K-nom 193.5
```

Output (`out/motion/motion.csv` + video):

| Kolom | Arti | Sifat |
|---|---|---|
| `z_rel` | jarak relatif thd awal track (1.0 → mengecil saat mendekat) | **eksak, vision-only, tanpa skala** |
| `z_m_assumed` | meter = `K_nom / w` | **asumsi skala** (identik `K/w`; nilai absolut tergantung `K_nom`) |
| `ttc_s` | time-to-contact = `w / (dw/dt)` detik | **murni vision-only**, tapi noisy (perlu smoothing) |

> Fisika: untuk 1 batang, `w(t)` saja sudah menentukan jarak relatif. Angka meter
> tak bisa lebih akurat dari `K/w` tanpa sumber skala metrik (Δ gerak drone).
> Yang benar-benar bebas-skala: `z_rel` dan `ttc_s`.

---

## 5. Recovery rosbag terpotong (batas FAT32 4 GB)

Sebagian bag terpotong tepat di 4 GB (`4294967295` byte) karena batas file FAT32
saat perekaman. ~4 GB pertama masih bisa diselamatkan:

```bash
sqlite3 <bag_korup>.db3 ".recover" | sqlite3 recovered/rec_xxx.db3
```
Lalu pakai `recovered/rec_xxx.db3` sebagai input di langkah 2–4.

> Cegah ke depan: rekam ke SD card **exFAT/ext4**, jangan FAT32.

---

## 6. Deploy ke drone (ROS2)

**Pakai python3 sistem yang punya `rclpy` (dari ROS2), BUKAN venv.**

File yang perlu di-copy ke drone: `ros2_batang_node.py`, `batang_estimator.py`,
`models/batang-best.pt`.

```bash
source /opt/ros/humble/setup.bash          # sesuaikan distro (humble/iron/jazzy)
sudo apt install ros-humble-cv-bridge
pip3 install ultralytics

# jalankan (cara langsung, paling simpel):
python3 ros2_batang_node.py --ros-args -p K:=193.5 -p model:=models/batang-best.pt
```

Node:
- **subscribe** `/camera_front` (sensor_msgs/Image, bgr8)
- **publish** `/batang/distance` (std_msgs/Float32, meter)
- **publish** `/batang/detection_image` (debug, bbox+jarak)

Cek:
```bash
ros2 topic echo /batang/distance
ros2 topic hz /batang/distance
```

**Real-time di Jetson** → ekspor TensorRT sekali:
```bash
yolo export model=models/batang-best.pt format=engine half=True
python3 ros2_batang_node.py --ros-args -p model:=models/batang-best.engine
```

Parameter node: `model`, `K`, `conf`, `select` (`largest`|`center`),
`image_topic`, `publish_debug_image`.

---

## Struktur file

| File | Fungsi |
|---|---|
| `extract_frames.py` | Ekstrak frame dari .db3 (parser CDR, tanpa ROS) |
| `detect_batang.py` | Deteksi + kalibrasi K + CSV jarak (analisis) |
| `batang_estimator.py` | Inti runtime: deteksi → pilih target → `Z=K/w` → filter median |
| `sim_realtime.py` | Simulasi real-time offline (video + CSV) |
| `motion_estimator.py` | Estimator vision-only: lacak 1 batang → `z_rel` + meter(asumsi) + TTC |
| `sim_motion.py` | Simulasi motion/looming (video + CSV) |
| `ros2_batang_node.py` | Node ROS2 untuk drone |
| `models/batang-best.pt` | Model YOLO (kelas: `batang_sawit`) |

## Status validasi

- Scene hover 3 m: estimasi **3.00 m** (std 0.13 m). ✅
- `K=193.5` transfer wajar antar-rekaman berbeda.
- Coverage deteksi ~50–58% (separuh frame tanpa deteksi).
- **Belum** ada GT multi-jarak → akurasi lintas rentang belum tervalidasi.
