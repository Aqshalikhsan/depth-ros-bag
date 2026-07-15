#!/usr/bin/env python3
"""Launch file deployment: menjalankan node estimator jarak batang di drone.

Menjalankan node estimator sebagai proses langsung (tanpa perlu build paket
colcon). Driver kamera dijalankan terpisah oleh pengguna; sesuaikan nama topik
lewat argumen image_topic bila berbeda dari /camera_front.

Contoh:
  ros2 launch batang_distance.launch.py K:=193.5
  ros2 launch batang_distance.launch.py image_topic:=/my_cam/image_raw K:=200
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_launch_description():
    args = [
        DeclareLaunchArgument('image_topic', default_value='/camera_front'),
        DeclareLaunchArgument('model', default_value=os.path.join(HERE, 'models/batang-best.pt')),
        DeclareLaunchArgument('K', default_value='193.5'),
        DeclareLaunchArgument('conf', default_value='0.35'),
        DeclareLaunchArgument('select', default_value='largest'),
    ]

    node = ExecuteProcess(
        cmd=[
            'python3', os.path.join(HERE, 'ros2_batang_node.py'), '--ros-args',
            '-p', ['image_topic:=', LaunchConfiguration('image_topic')],
            '-p', ['model:=', LaunchConfiguration('model')],
            '-p', ['K:=', LaunchConfiguration('K')],
            '-p', ['conf:=', LaunchConfiguration('conf')],
            '-p', ['select:=', LaunchConfiguration('select')],
        ],
        output='screen',
    )

    return LaunchDescription(args + [node])
