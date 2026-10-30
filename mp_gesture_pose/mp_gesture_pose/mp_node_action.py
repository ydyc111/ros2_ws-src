#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray
from cv_bridge import CvBridge
import cv2
import time
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class MpGesturePoseNode(Node):
    def __init__(self):
        super().__init__('mp_gesture_pose_node')

        # ============ 1. Tasks API 初始化 ============
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='/home/smp/models/hand_landmarker.task'),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,  # 为了防机械臂抽搐，只追踪1只手
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

        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

        # ============ 2. ROS 通信 ============
        self.bridge = CvBridge()
        self.sub_img = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        
        # 图像发布（用于可视化）
        self.pub_img = self.create_publisher(Image, '/mp_annotated', 10)
        # 手势发布
        self.pub_gesture = self.create_publisher(String, '/gesture_info', 10)
        # 姿态发布
        self.pub_pose = self.create_publisher(String, '/pose_info', 10)
        # ★ 新增：手部坐标与张开度发布（给机械臂同步用）
        self.pub_hand_pos = self.create_publisher(Float32MultiArray, '/hand_target_pos', 10)
        
        self.frame_timestamp_ms = 0
        self.get_logger().info('=== MediaPipe Tasks API Node (Action Sync) started ===')

    # ================= 几何分类逻辑 =================
    def classify_hand(self, landmarks):
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        fingers_up = [landmarks[t].y < landmarks[p].y for t, p in zip(tips, pips)]
        thumb_up = abs(landmarks[4].x - landmarks[0].x) > abs(landmarks[3].x - landmarks[0].x)

        if all(fingers_up) and thumb_up: return "OPEN_PALM"
        if not any(fingers_up) and not thumb_up: return "FIST"
        if fingers_up[1] and fingers_up[2] and not fingers_up[3] and not fingers_up[4]: return "VICTORY"
        if fingers_up[1] and not fingers_up[2] and not fingers_up[3] and not fingers_up[4]: return "POINTING"
        if thumb_up and not any(fingers_up): return "THUMBS_UP"
        return "UNKNOWN"

    def classify_pose(self, landmarks):
        left_hand_up = landmarks[15].y < landmarks[11].y
        right_hand_up = landmarks[16].y < landmarks[12].y
        if left_hand_up and right_hand_up: return "BOTH_ARMS_UP"
        if left_hand_up: return "LEFT_ARM_UP"
        if right_hand_up: return "RIGHT_ARM_UP"
        return "STANDING"

    # ================= 绘制逻辑 =================
    def draw_landmarks(self, frame, landmarks, connections, color_point, color_line):
        h, w = frame.shape[:2]
        for connection in connections:
            start_idx = connection.start
            end_idx = connection.end
            if start_idx >= len(landmarks) or end_idx >= len(landmarks):
                continue
            pt1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            pt2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            cv2.line(frame, pt1, pt2, color_line, 2)
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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        self.frame_timestamp_ms += 33
        h, w = frame.shape[:2]

        gesture_text = "NONE"
        pose_text = "NONE"

        # ---------- 1. 手部检测 ----------
        hand_results = self.hand_landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
        if hand_results.hand_landmarks:
            # 只取第一只手
            hand_landmarks = hand_results.hand_landmarks[0]
            handedness = hand_results.handedness[0][0].category_name
            
            # 手势分类
            gesture_text = self.classify_hand(hand_landmarks)
            
            # 绘制骨架
            self.draw_landmarks(frame, hand_landmarks, 
                                vision.HandLandmarksConnections.HAND_CONNECTIONS,
                                (0, 255, 0), (0, 200, 0))

            # ★ 计算手部目标位置与张开度
            wrist = hand_landmarks[0]
            
            # 计算张开度：五个指尖到手腕的平均距离
            tips = [4, 8, 12, 16, 20]
            dist = sum(math.sqrt((hand_landmarks[t].x - wrist.x)**2 + (hand_landmarks[t].y - wrist.y)**2) for t in tips) / 5.0
            
            # 发布坐标给控制端 [x, y, dist]
            pos_msg = Float32MultiArray()
            pos_msg.data = [float(wrist.x), float(wrist.y), float(dist)]
            self.pub_hand_pos.publish(pos_msg)
            
            # 画面标注手腕中心
            cx, cy = int(wrist.x * w), int(wrist.y * h)
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
            cv2.putText(frame, f'{handedness}:{gesture_text}', (cx - 40, cy - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # 显示张开度
            cv2.putText(frame, f'Dist: {dist:.2f}', (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # ---------- 2. 姿态检测 ----------
        pose_results = self.pose_landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
        if pose_results.pose_landmarks:
            pose_landmarks = pose_results.pose_landmarks[0]
            pose_text = self.classify_pose(pose_landmarks)
            
            self.draw_landmarks(frame, pose_landmarks,
                                vision.PoseLandmarksConnections.POSE_LANDMARKS,
                                (0, 0, 255), (255, 0, 0))

            cv2.putText(frame, pose_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # ---------- 3. 发布 ----------
        self.pub_gesture.publish(String(data=gesture_text))
        self.pub_pose.publish(String(data=pose_text))

        out_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        out_msg.header = msg.header
        self.pub_img.publish(out_msg)

        self.get_logger().info(f'Gesture: {gesture_text} | Pose: {pose_text}', throttle_duration_sec=1.0)

    def destroy_node(self):
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
