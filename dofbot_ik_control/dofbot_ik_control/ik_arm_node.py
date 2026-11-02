#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import time
import sys
import math

try:
    from Arm_Lib import Arm_Device
except ImportError:
    print("错误: 未找到 Arm_Lib，请确认已安装 DOFbot 官方库")
    sys.exit(1)

class DofbotIKController(Node):
    def __init__(self):
        super().__init__('dofbot_ik_controller')
        
        self.L1 = 0.085         
        self.L2 = 0.165         
        self.base_height = 0.10 
        
        self.fx = 498.65        
        self.fy = 500.33        
        self.cx_cam = 311.15    
        self.cy_cam = 250.89    
        self.REAL_WIDTH = 0.08  # 保持 0.08
        
        self.cam_offset_x = 0.0
        self.cam_offset_y = 0.0
        self.cam_offset_z = 0.0
        
        self.current_x, self.current_y, self.current_z = 0.0, 0.0, 0.20 
        self.last_s1, self.last_s2, self.last_s3 = 0.0, 0.0, 0.0
        self.first_msg = True
        
        self.arm = Arm_Device()
        time.sleep(0.5)
        
        # 启动时先让机械臂摆到安全姿势
        self.arm.Arm_serial_servo_write(1, 90, 1000)
        self.arm.Arm_serial_servo_write(2, 90, 1000)
        self.arm.Arm_serial_servo_write(3, 90, 1000)
        self.arm.Arm_serial_servo_write(4, 90, 1000)
        self.arm.Arm_serial_servo_write(5, 90, 1000)
        time.sleep(1.0)
        
        self.get_logger().info('=== DOFbot IK 控制节点已启动 (视野舒展版) ===')
        
        self.sub_target = self.create_subscription(
            Float32MultiArray, '/custom_target', self.target_callback, 10)

    def solve_ik(self, X, Y, Z):
        theta1 = math.atan2(Y, X)
        R = math.sqrt(X**2 + Y**2)
        S = Z - self.base_height
        D = math.sqrt(R**2 + S**2)
        
        if D > (self.L1 + self.L2 + 0.01):
            self.get_logger().warn(f'目标太远: {D*100:.0f}cm', throttle_duration_sec=2.0)
            return None
        if D < abs(self.L1 - self.L2) - 0.01:
            self.get_logger().warn(f'目标太近: {D*100:.0f}cm', throttle_duration_sec=2.0)
            return None
            
        cos3 = max(-1.0, min(1.0, (self.L1**2 + self.L2**2 - D**2) / (2 * self.L1 * self.L2)))
        theta3 = math.acos(cos3)
        
        cos_alpha = max(-1.0, min(1.0, (self.L1**2 + D**2 - self.L2**2) / (2 * self.L1 * D)))
        alpha = math.acos(cos_alpha)
        beta = math.atan2(S, R)
        
        theta2 = beta - alpha
        theta4 = -(theta2 - (math.pi - theta3))
        
        s1 = 90.0 + math.degrees(theta1)
        s2 = 180.0 - math.degrees(theta2)
        s3 = 270.0 - math.degrees(theta3)
        s5 = 90.0 - math.degrees(theta4)
        
        # ★ 放宽限幅：允许大臂小臂舒展，不再卡死
        s1 = max(0.0, min(180.0, s1))
        s2 = max(30.0, min(178.0, s2)) 
        s3 = max(45.0, min(135.0, s3)) 
        s5 = max(0.0, min(180.0, s5))
        
        return [s1, s2, s3, s5]

    def target_callback(self, msg):
        if len(msg.data) < 4: return
        
        cx, cy, w_px, h_px, conf = msg.data
        if w_px < 15: return
        
        Z_raw = (self.fx * self.REAL_WIDTH) / w_px
        
        # ★ 放宽距离拦截：只要不是极端离谱，都放行
        if Z_raw < 0.08 or Z_raw > 0.60:
            self.get_logger().warn(f'距离异常({Z_raw*100:.0f}cm)，已丢弃', throttle_duration_sec=1.0)
            return

        X_raw = (cx - self.cx_cam) * Z_raw / self.fx
        Y_raw = (cy - self.cy_cam) * Z_raw / self.fy
        
        X_base = X_raw + self.cam_offset_x
        Y_base = Y_raw + self.cam_offset_y
        Z_base = Z_raw + self.cam_offset_z
        
        # 死区：放宽到 0.03，让底座能追
        if abs(X_base) < 0.03: X_base = 0.03 if X_base >= 0 else -0.03
        if abs(Y_base) < 0.03: Y_base = 0.03 if Y_base >= 0 else -0.03

        if self.first_msg:
            self.current_x, self.current_y, self.current_z = X_base, Y_base, Z_base
            self.first_msg = False
        else:
            alpha = 0.25 # 稍微提高响应速度
            self.current_x = self.current_x * (1 - alpha) + X_base * alpha
            self.current_y = self.current_y * (1 - alpha) + Y_base * alpha
            self.current_z = self.current_z * (1 - alpha) + Z_base * alpha
        
        target_angles = self.solve_ik(self.current_x, self.current_y, self.current_z)
        
        if target_angles:
            s1, s2, s3, s5 = target_angles
            
            # 死区保护：变动大于 2 度才发，防止抖动
            if (abs(s1 - self.last_s1) < 2.0 and 
                abs(s2 - self.last_s2) < 2.0 and 
                abs(s3 - self.last_s3) < 2.0):
                return
            
            self.arm.Arm_serial_servo_write(1, int(s1), 300)
            self.arm.Arm_serial_servo_write(2, int(s2), 300)
            self.arm.Arm_serial_servo_write(3, int(s3), 300)
            self.arm.Arm_serial_servo_write(4, 90, 300)
            self.arm.Arm_serial_servo_write(5, int(s5), 300)
            
            self.last_s1, self.last_s2, self.last_s3 = s1, s2, s3
            
            # ★ 补全了被截断的日志打印
            self.get_logger().info(
                f'IK追踪: X={self.current_x:.2f}m, Y={self.current_y:.2f}m, Z={self.current_z:.2f}m | '
                f'底座:{s1:.0f}°, 大臂:{s2:.0f}°, 小臂:{s3:.0f}°'
            )

def main(args=None):
    rclpy.init(args=args)
    node = DofbotIKController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
