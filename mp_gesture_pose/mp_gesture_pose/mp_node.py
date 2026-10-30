#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MpGesturePoseNode(Node):
    def __init__(self):
        super().__init__('mp_gesture_pose_node')

        # ============ 1. Tasks API 初始化 ============
        # 请确保提前将模型文件下载到了 ~/models/ 目录下
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='/home/smp/models/hand_landmarker.task'),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='/home/smp/models/pose_landmarker_full.task'),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 【关键】实例化检测器，必须在 __init__ 中完成，严禁在回调里重复创建
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        # ============ 2. ROS 通信 ============
        self.bridge = CvBridge()
        self.sub_img = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.pub_img = self.create_publisher(Image, '/mp_annotated', 10)
        self.pub_gesture = self.create_publisher(String, '/gesture_info', 10)
        self.pub_pose = self.create_publisher(String, '/pose_info', 10)
        
        # 用于生成单调递增的时间戳（毫秒）
        self.frame_timestamp_ms = 0

        self.get_logger().info('=== MediaPipe Tasks API Node started ===')

    # ================= 几何分类逻辑 =================
    def classify_hand(self, landmarks):
        """基于21个手部关键点判断手势"""
        # 索引：4=拇指尖, 8=食指尖, 12=中指尖, 16=无名指尖, 20=小指尖
        # 指节：3=拇指节, 6=食指节, 10=中指节, 14=无名指节, 18=小指节
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        
        # 指尖 y < 指节 y 表示手指伸直（图像坐标 y 向下）
        fingers_up = [landmarks[t].y < landmarks[p].y for t, p in zip(tips, pips)]
        # 拇指需要看 x 方向
        thumb_up = abs(landmarks[4].x - landmarks[0].x) > abs(landmarks[3].x - landmarks[0].x)

        if all(fingers_up) and thumb_up: return "OPEN_PALM"
        if not any(fingers_up) and not thumb_up: return "FIST"
        if fingers_up[1] and fingers_up[2] and not fingers_up[3] and not fingers_up[4]: return "VICTORY"
        if fingers_up[1] and not fingers_up[2] and not fingers_up[3] and not fingers_up[4]: return "POINTING"
        if thumb_up and not any(fingers_up): return "THUMBS_UP"
        return "UNKNOWN"

    def classify_pose(self, landmarks):
        """基于33个身体关键点判断姿态"""
        # 索引：11=左肩, 12=右肩, 15=左腕, 16=右腕
        left_hand_up = landmarks[15].y < landmarks[11].y
        right_hand_up = landmarks[16].y < landmarks[12].y
        if left_hand_up and right_hand_up: return "BOTH_ARMS_UP"
        if left_hand_up: return "LEFT_ARM_UP"
        if right_hand_up: return "RIGHT_ARM_UP"
        return "STANDING"

    # ================= 绘制逻辑（修复版） =================
    def draw_landmarks(self, frame, landmarks, connections, color_point, color_line):
        h, w = frame.shape[:2]
        # 画线
        for connection in connections:
            # 【修复】新版 API 中 connection 是对象，包含 .start 和 .end 属性
            start_idx = connection.start
            end_idx = connection.end
            
            # 边界保护，防止越界
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
                
            pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            cv2.line(frame, pt1, pt2, color_line, 2)
            
        # 画点
        for lm in landmarks:
            pt = (int(lm.x * w), int(lm.y * h))
            cv2.circle(frame, pt, 3, color_point, -1)

    # ================= 主回调 =================
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        # 转换颜色空间 (Tasks API 明确要求 SRGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # 【关键】时间戳必须单调递增 (毫秒)
        self.frame_timestamp_ms += 33  # 假设约 30fps，每帧增加 33ms
        h, w = frame.shape[:2]

        gesture_text = "NONE"
        pose_text = "NONE"

        # ---------- 1. 手部检测 ----------
        hand_results = self.hand_landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
        if hand_results.hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.hand_landmarks):
                gesture_text = self.classify_hand(hand_landmarks)
                handedness = hand_results.handedness[idx][0].category_name
                
                # 绘制手部骨架
                self.draw_landmarks(frame, hand_landmarks, 
                                    vision.HandLandmarksConnections.HAND_CONNECTIONS,
                                    (0, 255, 0), (0, 200, 0))

                # 在手腕处标注
                wrist = hand_landmarks[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, f'{handedness}:{gesture_text}', (cx - 40, cy - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ---------- 2. 姿态检测 ----------
        pose_results = self.pose_landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
        if pose_results.pose_landmarks:
            for pose_landmarks in pose_results.pose_landmarks:
                pose_text = self.classify_pose(pose_landmarks)
                
                # 绘制姿态骨架
                self.draw_landmarks(frame, pose_landmarks,
                                    vision.PoseLandmarksConnections.POSE_LANDMARKS,
                                    (0, 0, 255), (255, 0, 0))

                # 左上角标注姿态
                cv2.putText(frame, pose_text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # ---------- 3. 发布 ----------
        self.pub_gesture.publish(String(data=gesture_text))
        self.pub_pose.publish(String(data=pose_text))

        out_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        out_msg.header = msg.header
        self.pub_img.publish(out_msg)

        # 降低日志打印频率，防止刷屏卡顿
        self.get_logger().info(f'Gesture: {gesture_text} | Pose: {pose_text}', throttle_duration_sec=1.0)

    def destroy_node(self):
        # 释放 C++ 底层资源
        self.hand_landmarker.close()
        self.pose_landmarker.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MpGesturePoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
