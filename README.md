# Estimasi Jarak Drone–Batang Sawit (Monokular)

Estimasi jarak drone ke batang kelapa sawit dari satu kamera RGB (monokular),
tanpa depth sensor, stereo, maupun LiDAR. Deteksi batang memakai YOLO
(`models/batang-best.pt`, kelas `batang_sawit`); jarak dihitung secara geometri
dari lebar bounding box. Tidak menggunakan jaringan estimasi kedalaman.

## Rumus

Model kamera lubang jarum untuk objek berukuran nyata `W` (meter) pada jarak `Z`
(meter) yang terproyeksi selebar `w` piksel:

    w = f · W / Z            →      Z = f · W / w

Fokus `f` (piksel) dan lebar batang `W` sama-sama tidak diketahui namun konstan,
sehingga digabung menjadi satu konstanta `K = f · W`:

    Z = K / w

Turunan yang dipakai program:

| Besaran            | Rumus                    | Keterangan                                  |
|--------------------|--------------------------|---------------------------------------------|
| Kalibrasi K        | `K = Z_gt · w_gt`        | dari satu scene ber–ground truth            |
| Jarak metrik       | `Z = K / w`              | butuh K (butuh GT atau nilai asumsi)        |
| Jarak relatif      | `Z_rel = w_ref / w`      | tanpa skala; 1.0 di awal, mengecil saat dekat |
| Time-to-contact    | `TTC = w / (dw/dt)`      | detik; tanpa skala                          |

Catatan: monokular pasif bersifat ambigu skala. Nilai dalam meter selalu
memerlukan satu referensi metrik (GT jarak, atau `K` asumsi). Jarak relatif dan
TTC tidak memerlukannya.

## Instalasi

    sudo apt install -y python3-opencv ffmpeg sqlite3
    pip3 install --user "numpy<2" ultralytics

`numpy<2` diperlukan agar kompatibel dengan OpenCV dari apt. Torch/CUDA diasumsikan
sudah terpasang. Verifikasi:

    python3 -c "import cv2, numpy, torch, ultralytics"

## Data

Rekaman berupa ROS2 bag (`.db3`) dengan satu topik `/camera_front`
(`sensor_msgs/Image`, `bgr8`, 640x360, ~30 fps). Program membaca `.db3` langsung
via SQLite tanpa memerlukan ROS terpasang.

## Penggunaan

### Kasus A — ground truth diketahui

Kalibrasi `K` dari satu frame diam yang jaraknya diketahui, lalu estimasi metrik.

    # 1. Ekstrak frame untuk menentukan frame kalibrasi
    python3 extract_frames.py <bag> -o frames/ --step 30

    # 2. Kalibrasi (mis. frame 2400 berjarak 3.0 m) + estimasi seluruh rekaman
    python3 detect_batang.py <bag> -o out/ --step 15 --calib-frame 2400 --calib-dist 3.0

    # Jika K sudah diketahui dari kalibrasi sebelumnya:
    python3 detect_batang.py <bag> -o out/ --step 15 --K 193.5

Keluaran: `out/detections.csv` (kolom `Z_est_m`) dan `out/vis/` bila `--save-vis`.

Simulasi real-time (memutar rekaman melalui estimator, menghasilkan video):

    python3 sim_realtime.py <bag> -o out/sim/ --every 5 --K 193.5

Menghasilkan `out/sim/realtime_h264.mp4` dan `realtime.csv`.

Nilai kalibrasi saat ini: `K ≈ 193.5` (dari scene hover 3.0 m; lebar batang 64.5 px).

### Kasus B — ground truth tidak diketahui

Melacak satu batang dan memakai pertumbuhan lebar bbox. Keluaran utama adalah
jarak relatif dan TTC (keduanya tanpa skala); nilai meter tetap ditampilkan
dengan skala asumsi `K_nom` dan diberi label `*`.

    python3 sim_motion.py <bag> -o out/motion/ --every 5 --K-nom 193.5

Keluaran `out/motion/motion.csv`:

- `z_rel`        — jarak relatif terhadap awal pelacakan (eksak, tanpa skala)
- `z_m_assumed`  — meter dengan skala asumsi (nilai absolut bergantung `K_nom`)
- `ttc_s`        — time-to-contact (detik, tanpa skala)

## Deployment ROS2 (di drone)

Gunakan `python3` sistem yang menyertakan `rclpy` (dari instalasi ROS2), bukan
virtualenv terpisah.

Berkas yang diperlukan di drone: `ros2_batang_node.py`, `batang_estimator.py`,
`models/batang-best.pt`.

    source /opt/ros/humble/setup.bash        # sesuaikan distro
    sudo apt install ros-humble-cv-bridge
    pip3 install ultralytics

    python3 ros2_batang_node.py --ros-args -p K:=193.5 -p model:=models/batang-best.pt

Node:

- subscribe : `/camera_front` (`sensor_msgs/Image`, bgr8)
- publish    : `/batang/distance` (`std_msgs/Float32`, meter)
- publish    : `/batang/detection_image` (debug, bounding box + jarak)

Verifikasi:

    ros2 topic echo /batang/distance
    ros2 topic hz /batang/distance

Untuk kinerja real-time pada Jetson, ekspor model ke TensorRT sekali:

    yolo export model=models/batang-best.pt format=engine half=True
    # lalu jalankan dengan -p model:=models/batang-best.engine

## Recovery bag terpotong

Sebagian rekaman terpotong tepat pada 4 GB (batas berkas FAT32 saat perekaman).
Bagian sebelum titik potong masih dapat diselamatkan:

    sqlite3 <bag_rusak>.db3 ".recover" | sqlite3 recovered/rec.db3

Hasilnya dipakai sebagai input pada perintah di atas. Untuk mencegah hal serupa,
rekam ke media exFAT atau ext4.

## Struktur berkas

| Berkas                | Fungsi                                                        |
|-----------------------|--------------------------------------------------------------|
| `extract_frames.py`   | Ekstraksi frame dari `.db3` (parser CDR, tanpa ROS)          |
| `detect_batang.py`    | Deteksi, kalibrasi K, estimasi metrik (batch)                |
| `batang_estimator.py` | Inti runtime: deteksi → `Z = K/w` → filter median            |
| `sim_realtime.py`     | Simulasi real-time metode metrik (video + CSV)               |
| `motion_estimator.py` | Estimator tanpa skala: `z_rel` + TTC + meter asumsi          |
| `sim_motion.py`       | Simulasi metode motion (video + CSV)                         |
| `ros2_batang_node.py` | Node ROS2 untuk drone                                        |
| `models/batang-best.pt` | Model YOLO deteksi batang                                  |

## Batasan

- Metode `K/w` mengasumsikan diameter batang relatif seragam.
- Lensa wide/fisheye belum dikoreksi; estimasi di tepi frame cenderung bias.
- Deteksi berhasil pada ~50–58% frame (sisanya batang terhalang, di tepi, atau jauh).
- Akurasi lintas jarak belum divalidasi terhadap ground truth berjenjang.
