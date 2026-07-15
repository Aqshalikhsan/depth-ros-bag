# Estimasi Jarak Drone–Batang Sawit (Monokular)

Estimasi jarak drone ke batang kelapa sawit dari satu kamera RGB (monokular),
tanpa depth sensor, stereo, maupun LiDAR. Deteksi batang memakai YOLO
(`models/batang-best.pt`, kelas `batang_sawit`); jarak dihitung secara geometri
dari lebar bounding box. Tidak menggunakan jaringan estimasi kedalaman.

## Demonstrasi

Estimasi jarak saat drone mendekati batang (simulasi dari rosbag, `Z = K/w`):

| | |
|---|---|
| ![Rekaman 11:46](out/sim_1146/approach.gif) | ![Rekaman 11:49](out/sim_1149/approach.gif) |
| ![Rekaman 11:53](out/sim_1153/approach.gif) | ![Rekaman 12:03](out/sim_12_03_27/approach_contoh.gif) |

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

## Deployment ROS2 (drone, kamera live)

Bagian ini untuk operasi nyata di drone dengan kamera langsung, bukan pemutaran
rosbag. Node estimator bekerja pada topik kamera live; rosbag hanya dipakai untuk
`sim_realtime.py`/`sim_motion.py` (uji offline).

Alur data:

    [driver kamera] --/camera_front--> [ros2_batang_node] --/batang/distance--> [konsumen]
                        (Image bgr8)                          (Float32, meter)     (avoidance / FC)

Driver kamera (usb_cam, v4l2_camera, gscam, atau node sendiri) dijalankan terpisah
dan harus mem-publish topik `sensor_msgs/Image`. Node ini yang mengonsumsinya.

Prasyarat. Gunakan `python3` sistem yang menyertakan `rclpy` (dari instalasi
ROS2), bukan virtualenv terpisah. Salin ke drone: `ros2_batang_node.py`,
`batang_estimator.py`, `batang_distance.launch.py`, `models/batang-best.pt`.

    source /opt/ros/humble/setup.bash        # sesuaikan distro
    sudo apt install ros-humble-cv-bridge
    pip3 install ultralytics

Menjalankan (pilih salah satu):

    # a. Langsung
    python3 ros2_batang_node.py --ros-args -p K:=193.5

    # b. Via launch (parameter lebih ringkas)
    ros2 launch batang_distance.launch.py K:=193.5

Bila topik kamera drone bukan `/camera_front`, arahkan ulang:

    ros2 launch batang_distance.launch.py image_topic:=/kamera/image_raw K:=193.5

Antarmuka node:

- subscribe : `image_topic` (default `/camera_front`, `sensor_msgs/Image` bgr8)
- publish    : `/batang/distance` (`std_msgs/Float32`, meter)
- publish    : `/batang/detection_image` (debug: bounding box + jarak)

Parameter: `image_topic`, `model`, `K`, `conf`, `select` (`largest`|`center`),
`publish_debug_image`.

Verifikasi saat drone menyala:

    ros2 topic list                     # pastikan topik kamera tampil
    ros2 topic echo /batang/distance    # nilai meter keluar saat batang terlihat
    ros2 topic hz /batang/distance      # laju publikasi

Kinerja real-time pada Jetson: ekspor model ke TensorRT sekali, lalu jalankan
dengan engine tersebut.

    yolo export model=models/batang-best.pt format=engine half=True
    python3 ros2_batang_node.py --ros-args -p model:=models/batang-best.engine

Alternatif `ros2 run <paket> <node>`: bungkus berkas menjadi paket ament_python
(`package.xml` + `setup.py` dengan entry point), lalu `colcon build`. Cara langsung
dan launch di atas tidak memerlukan build.

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
| `ros2_batang_node.py` | Node ROS2 untuk drone (kamera live)                          |
| `batang_distance.launch.py` | Launch file deployment (jalankan node + parameter)     |
| `models/batang-best.pt` | Model YOLO deteksi batang                                  |

