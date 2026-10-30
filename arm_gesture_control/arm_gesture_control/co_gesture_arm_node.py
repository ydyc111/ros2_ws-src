#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32MultiArray
import time
import threading
import sys

try:
    from Arm_Lib import Arm_Device
except ImportError:
    print("错误: 未找到 Arm_Lib，请确认已安装 DOFbot 官方库")
    sys.exit(1)

class CoGestureArmController(Node):
    def __init__(self):
        super().__init__('co_gesture_arm_controller')
        
        # ==================== 机械臂限制与参数 ====================
        self.SERVO_MIN = 10
        self.SERVO_MAX = 170
        self.SEND_INTERVAL = 0.05
        self.MOVE_TIME = 100       # 运动时间短一点，配合高频发送实现平滑跟随
        
        # 初始化机械臂
        self.arm = Arm_Device()
        time.sleep(0.5)
        
        # ==================== 方向反转开关 ====================
        # 如果机械臂底座左右转动和你的手是反的，把这里改成 False 试试
        self.REVERSE_X = True   
        # 如果机械臂大臂上下抬起和你的手是反的，把这里改成 False 试试
        self.REVERSE_Y = True   
        # ======================================================

        # 全局状态
        self.target_angles = [90] * 7
        self.last_sent = [None] * 7
        self.lock = threading.Lock()
        self.running = True
        self.is_moving = False  # 用于打断同步，执行全局复位等动作
        
        # 低通滤波初始值
        self.smooth_x = 0.5
        self.smooth_y = 0.5
        self.smooth_dist = 0.1
        
        # 启动后台平滑发送线程
        self.sender_thread = threading.Thread(target=self.sender_loop, daemon=True)
        self.sender_thread.start()
        
        # ==================== ROS 通信 ====================
        # 订阅手部实时坐标（核心，用于同步跟随）
        self.sub_hand = self.create_subscription(
            Float32MultiArray, '/hand_target_pos', self.hand_pos_callback, 10)
            
        # 订阅手势信息（用于特殊手势打断，比如比耶复位）
        self.sub_gesture = self.create_subscription(
            String, '/gesture_info', self.gesture_callback, 10)
            
        self.get_logger().info('=== 机械臂动作同步节点 (co_gesture_arm_node) 已启动 ===')

    def clamp(self, v):
        return max(self.SERVO_MIN, min(self.SERVO_MAX, int(v)))

    def set_target(self, sid, angle):
        with self.lock:
            self.target_angles[sid] = self.clamp(angle)

    def sender_loop(self):
        """后台线程：平滑向底层硬件发送角度指令"""
        while self.running and rclpy.ok():
            time.sleep(self.SEND_INTERVAL)
            with self.lock:
                snapshot = list(self.target_angles)
            for sid in range(1, 7):
                ang = snapshot[sid]
                if self.last_sent[sid] != ang:
                    try:
                        self.arm.Arm_serial_servo_write(sid, ang, self.MOVE_TIME)
                        self.last_sent[sid] = ang
                    except Exception as e:
                        self.get_logger().error(f"发送舵机{sid}失败: {e}")

    def hand_pos_callback(self, msg):
        """实时处理手部坐标，映射到机械臂关节"""
        if self.is_moving or len(msg.data) < 3:
            return
            
        x, y, dist = msg.data
        
        # 1. 低通滤波 (Alpha 滤波) - 防止画面抖动导致机械臂抽搐
        alpha = 0.2  # 越小越平滑，但延迟越大
        self.smooth_x = self.smooth_x * (1 - alpha) + x * alpha
        self.smooth_y = self.smooth_y * (1 - alpha) + y * alpha
        self.smooth_dist = self.smooth_dist * (1 - alpha) + dist * alpha
        
        # 2. 坐标映射
        # --- X 轴映射 (底座 Servo 1) ---
        if self.REVERSE_X:
            target_servo1 = (1.0 - self.smooth_x) * 180
        else:
            target_servo1 = self.smooth_x * 180
        
        # --- Y 轴映射 (大臂 Servo 2) ---
        if self.REVERSE_Y:
            target_servo2 = (1.0 - self.smooth_y) * 180
        else:
            target_servo2 = self.smooth_y * 180
            
        # 估算小臂 Servo 3 (根据手部高低做简单联动)
        target_servo3 = 90 + (0.5 - self.smooth_y) * 90
        
        # 手掌张开度映射到夹爪 Servo 6 (0.05 闭合, 0.15 张开)
        target_servo6 = (1.0 - (self.smooth_dist / 0.2)) * 150
        target_servo6 = max(30, min(150, target_servo6))

        # 3. 更新目标角度
        self.set_target(1, target_servo1)
        self.set_target(2, target_servo2)
        self.set_target(3, target_servo3)
        self.set_target(6, target_servo6)

    def gesture_callback(self, msg):
        """特殊手势打断机制"""
        gesture = msg.data
        if gesture == "VICTORY":
            self.get_logger().info('触发复位手势 (VICTORY)，正在归中...')
            self.is_moving = True  # 暂停跟随
            
            for i in range(1, 7):
                self.set_target(i, 90)
            
            time.sleep(1.5)  # 等待机械臂归中完成
            self.is_moving = False

    def destroy_node(self):
        self.running = False
        time.sleep(0.1)
        try:
            # 退出时让舵机断电，防止憋舵发热
            self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        except:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CoGestureArmController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
