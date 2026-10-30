#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from yolo_msgs.msg import DetectionArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from collections import defaultdict, deque


class FilterYoloNode(Node):
    def __init__(self):
        super().__init__('filter_yolo_node')

        # ============================================================
        # 一、YOLO 相关参数（COCO 80 类全开）
        # ============================================================
        self.target_classes = {
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
        }

        # 相机内参
        self.K = np.array([
            498.64946211, 0.0, 311.15160854,
            0.0, 500.33058382, 250.88784246,
            0.0, 0.0, 1.0
        ]).reshape(3, 3)
        self.D = np.array([
            -0.448831885, 0.231528364,
            -0.000022493, -0.003007713, -0.067148590
        ])

        # 80 类物体实际宽度（米），用于单目测距
        self.REAL_WIDTH = {
            'person': 0.45,
            'bicycle': 0.60, 'car': 1.80, 'motorcycle': 0.80,
            'airplane': 30.0, 'bus': 2.50, 'train': 3.00, 'truck': 2.40, 'boat': 3.00,
            'traffic light': 0.30, 'fire hydrant': 0.25,
            'stop sign': 0.60, 'parking meter': 0.20, 'bench': 1.20,
            'bird': 0.15, 'cat': 0.20, 'dog': 0.30, 'horse': 0.60,
            'sheep': 0.40, 'cow': 0.60, 'elephant': 2.00, 'bear': 0.80,
            'zebra': 0.60, 'giraffe': 0.80,
            'backpack': 0.30, 'umbrella': 0.80, 'handbag': 0.25,
            'tie': 0.10, 'suitcase': 0.45,
            'frisbee': 0.25, 'skis': 0.15, 'snowboard': 0.25,
            'sports ball': 0.22, 'kite': 0.60, 'baseball bat': 0.07,
            'baseball glove': 0.20, 'skateboard': 0.20,
            'surfboard': 0.50, 'tennis racket': 0.27,
            'bottle': 0.07, 'wine glass': 0.08, 'cup': 0.08,
            'fork': 0.03, 'knife': 0.03, 'spoon': 0.03, 'bowl': 0.15,
            'banana': 0.20, 'apple': 0.08, 'sandwich': 0.12,
            'orange': 0.08, 'broccoli': 0.15, 'carrot': 0.05,
            'hot dog': 0.15, 'pizza': 0.30, 'donut': 0.10, 'cake': 0.20,
            'chair': 0.45, 'couch': 1.80, 'potted plant': 0.30,
            'bed': 1.50, 'dining table': 1.20, 'toilet': 0.40,
            'tv': 0.90, 'laptop': 0.35, 'mouse': 0.06,
            'remote': 0.05, 'keyboard': 0.45, 'cell phone': 0.07,
            'microwave': 0.50, 'oven': 0.60, 'toaster': 0.30,
            'sink': 0.60, 'refrigerator': 0.70,
            'book': 0.15, 'clock': 0.25, 'vase': 0.15,
            'scissors': 0.10, 'teddy bear': 0.25, 'hair drier': 0.15, 'toothbrush': 0.03,
        }
        # 兜底宽度
        self.DEFAULT_WIDTH = 0.30

        # ============================================================
        # 二、形状检测相关参数
        # ============================================================
        self.enable_shape = False          # 形状检测总开关
        self.min_area = 800               # 最小轮廓面积
        self.approx_eps = 0.04            # 多边形逼近精度
        self.circ_thresh = 0.75           # 圆度阈值

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

        self.display_colors = {
            'red':     (0, 0, 255), 'orange':  (0, 128, 255),
            'yellow':  (0, 255, 255), 'green':   (0, 200, 0),
            'blue':    (255, 100, 0), 'purple':  (255, 0, 180),
            'pink':    (200, 100, 255), 'unknown': (200, 200, 200),
        }

        # ============================================================
        # 三、通用状态
        # ============================================================
        self.bridge = CvBridge()
        self.latest_detections = None
        self.latest_det_time = 0.0
        self.max_det_age = 0.5
        self.tracks = defaultdict(lambda: deque(maxlen=30))

        # ============================================================
        # 四、订阅 / 发布
        # ============================================================
        self.sub_img = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        self.sub_det = self.create_subscription(
            DetectionArray, '/detections', self.detection_callback, 10)

        self.pub_img = self.create_publisher(Image, '/filtered_image', 10)
        self.pub_info = self.create_publisher(String, '/shape_info', 10)

        self.get_logger().info(
            f'=== Filter + 3D + Tracking + Shape node started. '
            f'80 Classes enabled ===')

    # ================================================================
    # YOLO 相关方法
    # ================================================================
    def detection_callback(self, msg):
        self.latest_detections = msg
        self.latest_det_time = time.time()

    def pixel_to_3d(self, u, v, w_px, real_w):
        Z = (self.K[0, 0] * real_w) / w_px
        pts = np.array([[[u, v]]], dtype=np.float32)
        x_n, y_n = cv2.undistortPoints(pts, self.K, self.D)[0, 0]
        return x_n * Z, y_n * Z, Z

    def estimate_velocity(self, track_id):
        tr = self.tracks[track_id]
        if len(tr) < 2:
            return 0.0, 0.0, 0.0
        x0, y0, z0, t0 = tr[-2]
        x1, y1, z1, t1 = tr[-1]
        dt = t1 - t0
        if dt < 1e-3:
            return 0.0, 0.0, 0.0
        return (x1 - x0) / dt, (y1 - y0) / dt, (z1 - z0) / dt

    # ================================================================
    # 形状 + 颜色相关方法
    # ================================================================
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

    def detect_color(self, hsv, shape_mask):
        shape_area = cv2.countNonZero(shape_mask)
        if shape_area == 0:
            return None

        best_color, best_overlap = None, 0
        for name, ranges in self.color_ranges.items():
            color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                color_mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            overlap = cv2.countNonZero(
                cv2.bitwise_and(color_mask, shape_mask))
            if overlap > best_overlap:
                best_color, best_overlap = name, overlap

        if best_overlap / shape_area > 0.30:
            return best_color
        return None

    def draw_shape_info(self, frame, cx, cy, w, h,
                        shape_name, color_name, area):
        disp = self.display_colors.get(color_name, (200, 200, 200))
        cv2.circle(frame, (cx, cy), 5, disp, -1)

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

        bx = cx - box_w // 2
        by = cy - h // 2 - box_h - 10
        if by < 0:
            by = cy + h // 2 + 10
        if bx < 0:
            bx = 5
        if bx + box_w > frame.shape[1]:
            bx = frame.shape[1] - box_w - 5

        overlay = frame.copy()
        cv2.rectangle(overlay, (bx, by),
                      (bx + box_w, by + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (bx, by),
                      (bx + box_w, by + box_h), disp, 1)

        cv2.putText(frame, line1, (bx + 8, by + thh1 + 4),
                    font, fs1, disp, th1)
        cv2.putText(frame, line2, (bx + 8, by + thh1 + thh2 + 10),
                    font, fs2, (255, 255, 255), th2)
        cv2.putText(frame, line3, (bx + 8, by + thh1 + thh2 * 2 + 16),
                    font, fs2, (200, 200, 200), th2)

    # ================================================================
    # 主回调
    # ================================================================
    def image_callback(self, img_msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        now = time.time()

        # ============================================================
        # 第一部分：YOLO 80类 检测框 + 3D + 跟踪 ID
        # ============================================================
        dets_fresh = (self.latest_detections is not None and
                      now - self.latest_det_time < self.max_det_age)

        if dets_fresh:
            targets = []

            for det in self.latest_detections.detections:
                if det.class_name not in self.target_classes:
                    continue

                cx = det.bbox.center.position.x
                cy = det.bbox.center.position.y
                w  = det.bbox.size.x
                h  = det.bbox.size.y

                real_w = self.REAL_WIDTH.get(det.class_name,
                                             self.DEFAULT_WIDTH)
                if w <= 1:
                    continue

                X, Y, Z = self.pixel_to_3d(cx, cy, w, real_w)
                tid = det.id if det.id else '?'
                self.tracks[tid].append((X, Y, Z, now))
                vx, vy, vz = self.estimate_velocity(tid)

                targets.append({
                    'det': det, 'id': tid,
                    'cx': cx, 'cy': cy, 'w': w, 'h': h,
                    'X': X, 'Y': Y, 'Z': Z,
                    'vx': vx, 'vy': vy, 'vz': vz,
                })

            # 找最近目标
            nearest_id = None
            if targets:
                nearest = min(targets, key=lambda t: t['Z'])
                nearest_id = nearest['id']

            # 画 YOLO 框
            for t in targets:
                det = t['det']
                x1 = int(t['cx'] - t['w'] / 2)
                y1 = int(t['cy'] - t['h'] / 2)
                x2 = int(t['cx'] + t['w'] / 2)
                y2 = int(t['cy'] + t['h'] / 2)

                is_nearest = (t['id'] == nearest_id)
                color = (0, 0, 255) if is_nearest else (0, 255, 0)
                thickness = 3 if is_nearest else 2

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              color, thickness)

                label = f'[{t["id"]}] {det.class_name} {det.score:.2f}'
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, thickness)
                cv2.putText(frame, f'Z={t["Z"]:.2f}m',
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 255), 2)

                speed = (t['vx']**2 + t['vy']**2 + t['vz']**2) ** 0.5
                if speed > 0.01:
                    cv2.putText(frame, f'v={speed:.2f}m/s',
                                (x1, y2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 200, 0), 2)

                tag = ' <- NEAREST' if is_nearest else ''
                self.get_logger().info(
                    f'[YOLO] [{t["id"]:>2s}] {det.class_name:14s} '
                    f'X={t["X"]:+.3f} Y={t["Y"]:+.3f} Z={t["Z"]:.3f}  '
                    f'v={speed:.2f}m/s{tag}')

        # ============================================================
        # 第二部分：形状 + 颜色检测
        # ============================================================
        shape_results = []
        if self.enable_shape:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11, 2)
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(
                thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            thresh = cv2.morphologyEx(
                thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            idx = 0
            for c in contours:
                area = cv2.contourArea(c)
                if area < self.min_area:
                    continue

                shape_name, approx = self.detect_shape(c)
                if shape_name == 'unknown':
                    continue

                shape_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(shape_mask, [c], -1, 255, -1)

                color_name = self.detect_color(hsv, shape_mask) or 'unknown'

                M = cv2.moments(c)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                x, y, w, h = cv2.boundingRect(c)

                disp = self.display_colors.get(color_name,
                                               (200, 200, 200))

                # 形状轮廓
                cv2.drawContours(frame, [approx], -1, disp, 2)
                # 外接矩形（细线）
                cv2.rectangle(frame, (x, y), (x + w, y + h), disp, 1)
                # 信息框
                self.draw_shape_info(frame, cx, cy, w, h,
                                     shape_name, color_name, area)

                idx += 1
                line = (f'[{idx}] {color_name:8s} {shape_name:10s} '
                        f'center=({cx},{cy})  '
                        f'area={int(area)}  bbox={w}x{h}')
                shape_results.append(line)

            # 发布形状文本
            info_msg = String()
            info_msg.data = ('\n'.join(shape_results)
                             if shape_results else '(no shapes)')
            self.pub_info.publish(info_msg)

            if shape_results:
                self.get_logger().info(
                    '[SHAPE]\n' + info_msg.data)

        # ============================================================
        # 发布最终图像（YOLO + 形状 叠加）
        # ============================================================
        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_msg.header = img_msg.header
        self.pub_img.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FilterYoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
