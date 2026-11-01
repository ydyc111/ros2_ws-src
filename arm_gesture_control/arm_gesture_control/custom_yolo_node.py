#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort
import time

# ★ 新增 TF2 相关依赖
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class CustomYoloNode(Node):
    def __init__(self):
        super().__init__('custom_yolo_node')
        self.bridge = CvBridge()
        
        # ============ 初始化 ONNX 模型 ============
        model_path = '/home/smp/models/best.onnx'
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.get_logger().info(f'模型已加载: {model_path}')

        # ============ 相机内参与测距参数 ============
        # 注意：这里的参数请根据你之前标定的数据来填
        self.fx = 498.65
        self.fy = 500.33
        self.cx_cam = 311.15
        self.cy_cam = 250.89
        self.REAL_WIDTH = 0.06  # 铜葫芦的实际宽度（米），需根据实物微调

        # ============ 订阅与发布 ============
        self.sub_img = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        
        # 发布目标信息 [cx, cy, w, h, confidence]
        self.pub_target = self.create_publisher(Float32MultiArray, '/custom_target', 10)
        # 发布可视化图像
        self.pub_dbg_img = self.create_publisher(Image, '/custom_dbg_image', 10)

        # ★ 初始化 TF 广播者
        self.tf_broadcaster = TransformBroadcaster(self)
        self.get_logger().info('TF2 广播者已就绪，准备发布 gourd_link')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return
        start_time = time.time()
        orig_h, orig_w = frame.shape[:2]
        
        # 1. 预处理
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        
        # 2. 推理
        outputs = self.session.run(None, {self.input_name: img})[0]
        
        # 3. 后处理 (NMS)
        outputs = np.squeeze(outputs).T # [8400, 5]
        scores = np.max(outputs[:, 4:], axis=1)
        
        mask = scores > 0.5
        boxes = outputs[mask, :4]
        scores = scores[mask]
        
        target_msg = Float32MultiArray()
        
        if len(boxes) > 0:
            x = boxes[:, 0] - boxes[:, 2] / 2
            y = boxes[:, 1] - boxes[:, 3] / 2
            w = boxes[:, 2]
            h = boxes[:, 3]
            bboxes = np.column_stack((x, y, w, h)).tolist()
            
            indices = cv2.dnn.NMSBoxes(bboxes, scores.tolist(), score_threshold=0.5, nms_threshold=0.5)
            
            scale_x, scale_y = orig_w / 640, orig_h / 640
            if len(indices) > 0:
                # 只取置信度最高的一个
                best_idx = indices.flatten()[0]
                bx, by, bw, bh = bboxes[best_idx]
                
                # 还原到原图尺寸
                x1, y1 = int(bx * scale_x), int(by * scale_y)
                x2, y2 = int((bx + bw) * scale_x), int((by + bh) * scale_y)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                real_w = int(x2 - x1)
                real_h = int(y2 - y1)
                conf = float(scores[best_idx])
                
                # 发布 [cx, cy, w, h, conf]
                target_msg.data = [float(cx), float(cy), float(real_w), float(real_h), conf]
                self.pub_target.publish(target_msg)
                
                # ★ ==================== TF2 动态广播逻辑 ====================
                if real_w > 10: # 防止太小导致除零错误
                    # 1. 单目测距估算 Z (米)
                    Z = (self.fx * self.REAL_WIDTH) / real_w
                    
                    # 2. 反投影计算 X, Y (米)
                    X_cam = (cx - self.cx_cam) * Z / self.fx
                    Y_cam = (cy - self.cy_cam) * Z / self.fy
                    
                    # 3. 构造 TransformStamped 消息
                    t = TransformStamped()
                    t.header.stamp = self.get_clock().now().to_msg()
                    t.header.frame_id = 'camera_link'  # 父坐标系：相机
                    t.child_frame_id = 'gourd_link'    # 子坐标系：铜葫芦
                    
                    t.transform.translation.x = X_cam
                    t.transform.translation.y = Y_cam
                    t.transform.translation.z = Z
                    
                    # 假设目标无旋转（姿态朝向为默认）
                    t.transform.rotation.x = 0.0
                    t.transform.rotation.y = 0.0
                    t.transform.rotation.z = 0.0
                    t.transform.rotation.w = 1.0
                    
                    # 发送 TF！
                    self.tf_broadcaster.sendTransform(t)
                # ==============================================================
                
                # 画框可视化
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f'copper_gourd {conf:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        out_msg.header = msg.header
        self.pub_dbg_img.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CustomYoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
