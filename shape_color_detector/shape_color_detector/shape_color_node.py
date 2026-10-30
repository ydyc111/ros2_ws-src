#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
class ShapeColorDetector(Node):
    def __init__(self):
        super().__init__('shape_color_detector')
        # ============ 参数 ============
        self.min_area = 800          # 最小轮廓面积（像素²）
        self.approx_eps = 0.04       # 多边形逼近精度
        self.circ_thresh = 0.75      # 圆度阈值
        self.bridge = CvBridge()
        # 订阅原始图像
        self.sub_img = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        # 发布带标注的图像（画面显示）
        self.pub_img = self.create_publisher(Image, '/shape_image', 10)
        # 发布结构化文本
        self.pub_info = self.create_publisher(String, '/shape_info', 10)
        # ============ HSV 颜色范围 ============
        self.color_ranges = {
            'red':    [((0, 100, 100), (10, 255, 255)),
                       ((170, 100, 100), (180, 255, 255))],
            'orange': [((10, 100, 100), (25, 255, 255))],
            'yellow': [((25, 100, 100), (35, 255, 255))],
            'green':  [((35, 80, 80), (85, 255, 255))],
            'blue':   [((85, 80, 80), (130, 255, 255))],
            'purple': [((130, 80, 80), (160, 255, 255))],
            'pink':   [((160, 80, 80), (170, 255, 255))],
        }
        # 每个形状在画面上显示的颜色（BGR）
        self.display_colors = {
            'red':    (0, 0, 255),
            'orange': (0, 128, 255),
            'yellow': (0, 255, 255),
            'green':  (0, 200, 0),
            'blue':   (255, 100, 0),
            'purple': (255, 0, 180),
            'pink':   (200, 100, 255),
            'unknown': (200, 200, 200),
        }
        self.get_logger().info('=== Shape + Color detector started ===')
    # ==================== 形状识别 ====================
    def detect_shape(self, contour):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, self.approx_eps * peri, True)
        v = len(approx)
        if v == 3:
            return 'triangle', approx
        if v == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h) if h > 0 else 0
            if 0.90 <= ar <= 1.10:
                return 'square', approx
            return 'rectangle', approx
        if v == 5:
            return 'pentagon', approx
        if v == 6:
            return 'hexagon', approx
        area = cv2.contourArea(contour)
        if peri > 0:
            circ = 4 * np.pi * area / (peri * peri)
            if circ > self.circ_thresh:
                return 'circle', approx
        return 'unknown', approx
    # ==================== 颜色识别 ====================
    def detect_color(self, hsv, shape_mask):
        shape_area = cv2.countNonZero(shape_mask)
        if shape_area == 0:
            return None
        best_color, best_overlap = None, 0
        for name, ranges in self.color_ranges.items():
            color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                color_mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            overlap = cv2.countNonZero(cv2.bitwise_and(color_mask, shape_mask))
            if overlap > best_overlap:
                best_color, best_overlap = name, overlap
        if best_overlap / shape_area > 0.30:
            return best_color
        return None
    # ==================== 画面上标注 ====================
    def draw_shape_info(self, frame, cx, cy, w, h, shape_name, color_name, area):
        """
        在画面上画：
          - 中心点
          - 颜色 + 形状名（第 1 行）
          - 坐标 (x, y)（第 2 行）
          - 面积（第 3 行，可选）
        """
        # 该颜色对应的显示色，用于字体
        disp = self.display_colors.get(color_name, (200, 200, 200))
        # 中心点
        cv2.circle(frame, (cx, cy), 5, disp, -1)
        # 信息框的位置：优先放在中心点上方
        line1 = f'{color_name} {shape_name}'
        line2 = f'({cx}, {cy})'
        line3 = f'area={int(area)}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs1, fs2 = 0.6, 0.5
        th1, th2 = 2, 1
        (tw1, thh1), _ = cv2.getTextSize(line1, font, fs1, th1)
        (tw2, thh2), _ = cv2.getTextSize(line2, font, fs2, th2)
        (tw3, _), _    = cv2.getTextSize(line3, font, fs2, th2)
        box_w = max(tw1, tw2, tw3) + 16
        box_h = thh1 + thh2 * 2 + 20
        # 计算信息框左上角位置
        bx = cx - box_w // 2
        by = cy - h // 2 - box_h - 10
        if by < 0:
            by = cy + h // 2 + 10
        if bx < 0:
            bx = 5
        if bx + box_w > frame.shape[1]:
            bx = frame.shape[1] - box_w - 5
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (bx, by), (bx + box_w, by + box_h),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        # 边框
        cv2.rectangle(frame,
                      (bx, by), (bx + box_w, by + box_h),
                      disp, 1)
        # 文字
        cv2.putText(frame, line1,
                    (bx + 8, by + thh1 + 4),
                    font, fs1, disp, th1)
        cv2.putText(frame, line2,
                    (bx + 8, by + thh1 + thh2 + 10),
                    font, fs2, (255, 255, 255), th2)
        cv2.putText(frame, line3,
                    (bx + 8, by + thh1 + thh2 * 2 + 16),
                    font, fs2, (200, 200, 200), th2)
    # ==================== 主回调 ====================
    def image_callback(self, img_msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return
        # ---------- 预处理 ----------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # ---------- 遍历所有轮廓 ----------
        results = []
        idx = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            shape_name, approx = self.detect_shape(c)
            if shape_name == 'unknown':
                continue
            # 形状 mask 用于颜色判断
            shape_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(shape_mask, [c], -1, 255, -1)
            color_name = self.detect_color(hsv, shape_mask) or 'unknown'
            # 中心坐标
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # 外接矩形
            x, y, w, h = cv2.boundingRect(c)
            # ---------- 画面标注 ----------
            disp = self.display_colors.get(color_name, (200, 200, 200))
            # 1. 形状轮廓
            cv2.drawContours(frame, [approx], -1, disp, 2)
            # 2. 外接矩形（虚线感：细线）
            cv2.rectangle(frame, (x, y), (x + w, y + h), disp, 1)
            # 3. 中心点 + 文字信息框
            self.draw_shape_info(frame, cx, cy, w, h,
                                 shape_name, color_name, area)
            # ---------- 文本结果 ----------
            idx += 1
            line = (f'[{idx}] {color_name:8s} {shape_name:10s} '
                    f'center=({cx},{cy})  '
                    f'area={int(area)}  bbox={w}x{h}')
            results.append(line)
        # ---------- 发布文本 ----------
        info_msg = String()
        if results:
            info_msg.data = '\n'.join(results)
            self.get_logger().info('\n' + info_msg.data)
        else:
            info_msg.data = '(no shapes)'
        self.pub_info.publish(info_msg)
        # ---------- 发布图像 ----------
        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_msg.header = img_msg.header
        self.pub_img.publish(out_msg)
def main(args=None):
    rclpy.init(args=args)
    node = ShapeColorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()
