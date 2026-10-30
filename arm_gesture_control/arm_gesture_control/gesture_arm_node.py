#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
import threading
import sys

try:
    from Arm_Lib import Arm_Device
except ImportError:
    print("错误: 未找到 Arm_Lib，请确认已安装 DOFbot 官方库")
    sys.exit(1)

class GestureArmController(Node):
    def __init__(self):
        super().__init__('arm_gesture_controller')
        
        # ==================== 用户配置区 (来自你的文档) ====================
        self.SERVO_MIN = 0
        self.SERVO_MAX = 180
        self.CENTER = 90
        self.SEND_INTERVAL = 0.05  # 舵机指令最小发送间隔
        self.MOVE_TIME = 500       # 单条指令的转动时间 (ms)
        self.NUM_SERVOS = 6
        # ====================================================================

        # 初始化机械臂
        self.arm = Arm_Device()
        time.sleep(0.5)
        
        # 全局状态（平滑控制核心）
        self.target_angles = [self.CENTER] * (self.NUM_SERVOS + 1)
        self.last_sent = [None] * (self.NUM_SERVOS + 1)
        self.lock = threading.Lock()
        self.running = True
        
        # 防抖状态
        self.last_gesture = "NONE"
        self.gesture_count = 0
        
        # 启动后台发送线程（你的代码里的核心亮点！）
        self.sender_thread = threading.Thread(target=self.sender_loop, daemon=True)
        self.sender_thread.start()
        
        # 订阅 MediaPipe 发布的话题
        self.sub_gesture = self.create_subscription(
            String, '/gesture_info', self.gesture_callback, 10)
        self.sub_pose = self.create_subscription(
            String, '/pose_info', self.pose_callback, 10)
            
        self.get_logger().info('=== 机械臂手势控制节点已启动 (平滑控制版) ===')

    def clamp(self, v):
        return max(self.SERVO_MIN, min(self.SERVO_MAX, int(v)))

    def set_target(self, sid, angle):
        with self.lock:
            self.target_angles[sid] = self.clamp(angle)

    def sender_loop(self):
        """后台平滑发送线程：只有角度变化时才发，避免堵塞串口"""
        while self.running and rclpy.ok():
            time.sleep(self.SEND_INTERVAL)
            with self.lock:
                snapshot = list(self.target_angles)
            for sid in range(1, self.NUM_SERVOS + 1):
                ang = snapshot[sid]
                if self.last_sent[sid] != ang:
                    try:
                        self.arm.Arm_serial_servo_write(sid, ang, self.MOVE_TIME)
                        self.last_sent[sid] = ang
                    except Exception as e:
                        self.get_logger().error(f"发送舵机{sid}失败: {e}")

    def gesture_callback(self, msg):
        gesture = msg.data
        
        # 防抖：只处理有效手势
        if gesture in ["UNKNOWN", "NONE"]:
            self.gesture_count = 0
            self.last_gesture = gesture
            return
            
        if gesture == self.last_gesture:
            self.gesture_count += 1
        else:
            self.last_gesture = gesture
            self.gesture_count = 1
            
        # 连续 5 帧识别到同一手势才触发
        if self.gesture_count == 2:
            self.execute_action(gesture)

    def execute_action(self, gesture):
        self.get_logger().info(f'触发手势: {gesture}')
        
        if gesture == "FIST":  # ✊ 握拳：夹爪闭合
            self.get_logger().info('-> 夹爪闭合')
            self.set_target(6, 150)
            
        elif gesture == "OPEN_PALM":  # 🖐 张开：夹爪张开
            self.get_logger().info('-> 夹爪张开')
            self.set_target(6, 30)
            
        elif gesture == "VICTORY":  # ✌️ 比耶：机械臂复位
            self.get_logger().info('-> 机械臂复位')
            for i in range(1, 7):
                self.set_target(i, 90)
                
        elif gesture == "POINTING":  # 👆 指天：抬起大臂
            self.get_logger().info('-> 抬起大臂')
            self.set_target(2, 120)
            self.set_target(3, 60)

    def pose_callback(self, msg):
        pose = msg.data
        if pose == "BOTH_ARMS_UP": # 🙌 双手举高：紧急停止/保持现状
            pass # 可以加上急停逻辑

    def destroy_node(self):
        self.running = False
        time.sleep(0.1)
        # 退出时让舵机失力
        try:
            self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        except:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = GestureArmController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
