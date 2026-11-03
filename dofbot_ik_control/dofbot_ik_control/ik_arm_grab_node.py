#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import time
import sys
import math
import threading
try:
    from Arm_Lib import Arm_Device
except ImportError:
    print("错误: 未找到 Arm_Lib，请确认已安装 DOFbot 官方库")
    sys.exit(1)
class DofbotIKController(Node):
    def __init__(self):
        super().__init__('dofbot_ik_controller')
        
        # ================= 1. 物理标定参数 =================
        self.L1 = 0.085         # 大臂长度 (8.5cm)
        self.L2 = 0.165         # 小臂长度 (16.5cm)
        self.base_height = 0.10 # 底座高度 (10cm)
        
        # ================= 2. 相机参数与安装偏差 =================
        self.fx = 498.65        
        self.fy = 500.33        
        self.cx_cam = 311.15    
        self.cy_cam = 250.89    
        self.REAL_WIDTH = 0.08  # 目标物实际宽度 (米)
        
        self.cam_offset_x = 0.0
        self.cam_offset_y = 0.0
        self.cam_offset_z = 0.0
        
        # ================= 3. 状态与防抖变量 =================
        self.current_x, self.current_y, self.current_z = 0.0, 0.0, 0.20 
        self.last_s1, self.last_s2, self.last_s3 = 0.0, 0.0, 0.0
        self.first_msg = True
        
        # ★ 新增：机器人状态机
        self.robot_state = 'FOLLOW'  # 状态：FOLLOW(跟随), GRABBING(抓取中)
        
        # ================= 4. 初始化硬件 =================
        self.arm = Arm_Device()
        time.sleep(0.5)
        
        # 启动时先让机械臂摆到安全姿势，夹爪张开
        self.arm.Arm_serial_servo_write(1, 90, 1000)
        self.arm.Arm_serial_servo_write(2, 90, 1000)
        self.arm.Arm_serial_servo_write(3, 90, 1000)
        self.arm.Arm_serial_servo_write(4, 90, 1000)
        self.arm.Arm_serial_servo_write(5, 90, 1000)
        self.arm.Arm_serial_servo_write(6, 30, 1000)  # 6号夹爪：30为张开
        time.sleep(1.0)
        
        self.get_logger().info('=== DOFbot IK 视觉抓取节点已启动 ===')
        
        # ================= 5. 订阅视觉话题 =================
        self.sub_target = self.create_subscription(
            Float32MultiArray, '/custom_target', self.target_callback, 10)
    def solve_ik(self, X, Y, Z):
        """几何解析法逆运动学解算"""
        theta1 = math.atan2(Y, X)
        R = math.sqrt(X**2 + Y**2)
        S = Z - self.base_height
        D = math.sqrt(R**2 + S**2)
        
        if D > (self.L1 + self.L2 + 0.01):
            return None
        if D < abs(self.L1 - self.L2) - 0.01:
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
        
        # 安全限幅 (防砸桌、防堵转)
        s1 = max(0.0, min(180.0, s1))
        s2 = max(45.0, min(160.0, s2)) 
        s3 = max(50.0, min(125.0, s3)) 
        s5 = max(0.0, min(180.0, s5))
        
        return [s1, s2, s3, s5]
    def target_callback(self, msg):
        if len(msg.data) < 4: return
        
        cx, cy, w_px, h_px, conf = msg.data
        if w_px < 15: return
        
        # 如果正在抓取，忽略视觉数据，专心执行抓取动作
        if self.robot_state != 'FOLLOW':
            return
            
        Z_raw = (self.fx * self.REAL_WIDTH) / w_px
        
        # 过滤极端异常的距离
        if Z_raw < 0.08 or Z_raw > 0.60:
            return
        X_raw = (cx - self.cx_cam) * Z_raw / self.fx
        Y_raw = (cy - self.cy_cam) * Z_raw / self.fy
        
        X_base = X_raw + self.cam_offset_x
        Y_base = Y_raw + self.cam_offset_y
        Z_base = Z_raw + self.cam_offset_z
        
        # 死区保护，防止底座在零点疯狂甩动
        if abs(X_base) < 0.03: X_base = 0.03 if X_base >= 0 else -0.03
        if abs(Y_base) < 0.03: Y_base = 0.03 if Y_base >= 0 else -0.03
        # 平滑滤波
        if self.first_msg:
            self.current_x, self.current_y, self.current_z = X_base, Y_base, Z_base
            self.first_msg = False
        else:
            alpha = 0.25 
            self.current_x = self.current_x * (1 - alpha) + X_base * alpha
            self.current_y = self.current_y * (1 - alpha) + Y_base * alpha
            self.current_z = self.current_z * (1 - alpha) + Z_base * alpha
        
        # ================= 核心触发逻辑：如果距离足够近，开启抓取 =================
        if self.current_z < 0.18 and w_px > 100:
            self.robot_state = 'GRABBING'
            self.get_logger().info('🎯 目标足够近，启动抓取线程！')
            threading.Thread(target=self.execute_grab).start()
            return
        
        # 正常跟随动作
        target_angles = self.solve_ik(self.current_x, self.current_y, self.current_z)
        if target_angles:
            s1, s2, s3, s5 = target_angles
            
            # 死区保护：角度变动大于 2 度才发
            if (abs(s1 - self.last_s1) < 2.0 and 
                abs(s2 - self.last_s2) < 2.0 and 
                abs(s3 - self.last_s3) < 2.0):
                return
            
            self.arm.Arm_serial_servo_write(1, int(s1), 300)
            self.arm.Arm_serial_servo_write(2, int(s2), 300)
            self.arm.Arm_serial_servo_write(3, int(s3), 300)
            self.arm.Arm_serial_servo_write(4, 90, 300)
            self.arm.Arm_serial_servo_write(5, int(s5), 300)
            self.arm.Arm_serial_servo_write(6, 30, 300)  # 确保移动时夹爪是张开的
            
            self.last_s1, self.last_s2, self.last_s3 = s1, s2, s3
            
            self.get_logger().info(
                f'跟随中: X={self.current_x:.2f}m, Y={self.current_y:.2f}m, Z={self.current_z:.2f}m | '
                f'底座:{s1:.0f}°, 大臂:{s2:.0f}°, 小臂:{s3:.0f}°'
            )
    def execute_grab(self):
        """抓取动作序列（在独立线程中运行，防止阻塞 ROS）"""
        # 1. 锁定当前坐标，算出最终抓取角度
        angles = self.solve_ik(self.current_x, self.current_y, self.current_z)
        if not angles:
            self.robot_state = 'FOLLOW'
            return
            
        s1, s2, s3, s5 = angles
        self.get_logger().info(f'锁定目标，移动到: 底座:{s1:.0f}°, 大臂:{s2:.0f}°, 小臂:{s3:.0f}°')
        
        # 2. 机械臂快速移动过去
        self.arm.Arm_serial_servo_write(1, int(s1), 400)
        self.arm.Arm_serial_servo_write(2, int(s2), 400)
        self.arm.Arm_serial_servo_write(3, int(s3), 400)
        self.arm.Arm_serial_servo_write(4, 90, 400)
        self.arm.Arm_serial_servo_write(5, int(s5), 400)
        
        # 3. 等待机械臂到位
        time.sleep(0.8)
        
        # 4. 夹爪闭合（150为闭合，夹住葫芦）
        self.get_logger().info('夹爪闭合！')
        self.arm.Arm_serial_servo_write(6, 150, 500)
        time.sleep(0.8)
        
        # 5. 机械臂大臂抬起，完成抓取悬空
        self.get_logger().info('向上抬起，抓取成功！')
        self.arm.Arm_serial_servo_write(2, max(45, int(s2 - 25)), 800)
        time.sleep(1.5)
        
        # 6. 张开夹爪，释放物体，回归跟随状态
        self.get_logger().info('松开夹爪，恢复跟随模式。')
        self.arm.Arm_serial_servo_write(6, 30, 500)
        time.sleep(0.5)
        
        self.robot_state = 'FOLLOW'
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
