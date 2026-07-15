#!/usr/bin/env python3
"""Node ROS2: estimasi jarak drone->batang sawit real-time (vision-only).

Subscribe : /camera_front  (sensor_msgs/Image, bgr8)
Publish   : /batang/distance  (std_msgs/Float32, meter; hanya saat valid)
            /batang/detection_image (sensor_msgs/Image, opsional, utk debug)

Jalankan (di companion computer drone, ROS2 sudah ter-source):
  python3 ros2_batang_node.py --ros-args -p K:=193.5 -p conf:=0.35

Catatan performa onboard: untuk real-time di Jetson, ekspor model ke TensorRT
  yolo export model=models/batang-best.pt format=engine half=True
lalu set param model:=models/batang-best.engine
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from ultralytics import YOLO

from batang_estimator import BatangDistanceEstimator


class BatangDistanceNode(Node):
    def __init__(self):
        super().__init__('batang_distance')
        self.declare_parameter('model', 'models/batang-best.pt')
        self.declare_parameter('K', 193.5)
        self.declare_parameter('conf', 0.35)
        self.declare_parameter('select', 'largest')
        self.declare_parameter('image_topic', '/camera_front')
        self.declare_parameter('publish_debug_image', True)

        gp = self.get_parameter
        model = YOLO(gp('model').value)
        self.est = BatangDistanceEstimator(
            model,
            K=float(gp('K').value),
            conf=float(gp('conf').value),
            select=gp('select').value,
        )
        self.bridge = CvBridge()
        self.pub_dist = self.create_publisher(Float32, '/batang/distance', 10)
        self.pub_dbg = None
        if gp('publish_debug_image').value:
            self.pub_dbg = self.create_publisher(Image, '/batang/detection_image', 10)

        self.create_subscription(Image, gp('image_topic').value, self.on_image, 10)
        self.get_logger().info('batang_distance node siap.')

    def on_image(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        r = self.est.update(img)

        if r['z'] is not None:
            self.pub_dist.publish(Float32(data=float(r['z'])))

        if self.pub_dbg is not None:
            import cv2
            det = r['detection']
            if det:
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            z = r['z']
            txt = f"{z:0.2f} m" if z is not None else "-- m"
            cv2.putText(img, txt, (8, img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            out = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            out.header = msg.header
            self.pub_dbg.publish(out)


def main():
    rclpy.init()
    node = BatangDistanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
