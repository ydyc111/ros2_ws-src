#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import time
import threading
import sys
try:
    from Arm_Lib import Arm_Device
except ImportError:
    print("错误: 未找到 Arm_Lib")
    sys.exit(1)
class CoCustomArmController(Node):
    def __init__(self):
        super().__init__('co_custom_arm_controller')
        
        # ==================== 硬件参数 ====================
        self.SERVO_MIN = 10
        self.SERVO_MAX = 170
        self.SEND_INTERVAL = 0.05
        self.MOVE_TIME = 150
        
        # 相机内参 (你之前标定的数据)
        self.fx = 498.65
        self.fy = 500.33
        self.cx_cam = 311.15
        self.cy_cam = 250.89
        
        # 铜葫芦的真实宽度 (米)，请用尺子量一下真实的铜葫芦宽度
        self.REAL_WIDTH = 0.06 
        
        # 初始化机械臂
        self.arm = Arm_Device()
        time.sleep(0.5)
        
        # 状态
        self.target_angles = [90] * 7
        self.last_sent = [None] * 7
        self.lock = threading.Lock()
        self.running = True
        self.is_moving = False
        
        # 滤波
        self.smooth_x = 0.0
        self.smooth_y = 0.0
        self.smooth_z = 0.5
        
        # 启动后台发送线程
        self.sender_thread = threading.Thread(target=self.sender_loop, daemon=True)
        self.sender_thread.start()
        
        # 订阅视觉节点的坐标
        self.sub_target = self.create_subscription(
            Float32MultiArray, '/custom_target', self.target_callback, 10)
            
        self.get_logger().info('=== 铜葫芦专属跟随节点已启动 ===')
    def clamp(self, v):
        return max(self.SERVO_MIN, min(self.SERVO_MAX, int(v)))
    def set_target(self, sid, angle):
        with self.lock:
            self.target_angles[sid] = self.clamp(angle)
    def sender_loop(self):
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
                    except Exception:
                        pass
    def target_callback(self, msg):
        if self.is_moving or len(msg.data) < 4:
            return
            
        cx, cy, w_px, h_px, conf = msg.data
        
        # 1. 单目测距计算 Z (米)
        if w_px < 10: return
        Z = (self.fx * self.REAL_WIDTH) / w_px
        
        # 2. 反投影计算 X, Y (米)
        X = (cx - self.cx_cam) * Z / self.fx
        Y = (cy - self.cy_cam) * Z / self.fy
        
        # 3. 低通滤波平滑
        alpha = 0.3
        self.smooth_x = self.smooth_x * (1 - alpha) + X * alpha
        self.smooth_y = self.smooth_y * (1 - alpha) + Y * alpha
        self.smooth_z = self.smooth_z * (1 - alpha) + Z * alpha
        
        # 4. 坐标映射 (需根据实际机械臂安装位置微调)
        # 假设机械臂底座在相机正下方，转动角度由 X 决定
        target_servo1 = 90 + (self.smooth_x * 150)
        
        # 大臂根据距离 Z 俯仰
        target_servo2 = 90 + (self.smooth_z - 0.5) * 100
        # 小臂联动
        target_servo3 = 90 - (self.smooth_z - 0.5) * 80
        
        # 夹爪逻辑：如果离得很近（Z < 0.25 米），闭合夹爪
        if self.smooth_z < 0.25:
            self.set_target(6, 150) # 闭合
            self.get_logger().info('距离过近，执行抓取！', throttle_duration_sec=1.0)
        else:
            self.set_target(6, 30)  # 张开
        # 限制范围并发送
        self.set_target(1, target_servo1)
        self.set_target(2, target_servo2)
        self.set_target(3, target_servo3)
        
        self.get_logger().info(f'目标坐标: X={self.smooth_x:.2f}, Y={self.smooth_y:.2f}, Z={self.smooth_z:.2f}', throttle_duration_sec=0.5)
    def destroy_node(self):
        self.running = False
        time.sleep(0.1)
        try:
            self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        except:
            pass
        super().destroy_node()
def main(args=None):
    rclpy.init(args=args)
    node = CoCustomArmController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()
