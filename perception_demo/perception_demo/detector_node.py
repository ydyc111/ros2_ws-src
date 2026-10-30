#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')
        self.bridge = CvBridge()
        # 订阅话题（节点内名称 /camera/image_raw，实际运行时会被重映射）
        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        # 可选：发布检测结果（省略，仅打印信息）
        self.get_logger().info('Detector node started.')
    def image_callback(self, msg):
        try:
            # 注意：相机发布的编码是 rgb8，这里直接转为 OpenCV BGR 格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Conversion error: {e}')
            return
        # 转换为 HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        # 红色阈值（两个区间，因为红色在 HSV 中分布在 0 和 180 附近）
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:   # 忽略太小区域
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            center_x = x + w//2
            center_y = y + h//2
            self.get_logger().info(f'Detected red object at center: ({center_x}, {center_y}), area: {area}')
        # （可选）显示图像用于调试，但 Jetson 无显示器时请注释掉，这里取消注释
        cv2.imshow('view', cv_image)
        cv2.waitKey(1)
def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()
