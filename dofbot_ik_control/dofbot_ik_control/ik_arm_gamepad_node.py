#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
import time, sys, math
from Arm_Lib import Arm_Device

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
        self.REAL_WIDTH = 0.08  
        
        # 初始坐标与偏移
        self.vis_x, self.vis_y, self.vis_z = 0.0, 0.0, 0.25
        self.off_x, self.off_y, self.off_z = 0.0, 0.0, 0.0
        
        # 摇杆增量速度 (推满摇杆时，每次循环移动多少米)
        self.speed = 0.003 
        
        self.last_s1, self.last_s2, self.last_s3 = 0.0, 0.0, 0.0
        self.arm = Arm_Device()
        time.sleep(0.5)
        self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 30, 1000) # 启动张爪
        
        self.sub_vis = self.create_subscription(Float32MultiArray, '/custom_target', self.vis_callback, 10)
        self.sub_joy = self.create_subscription(Float32MultiArray, '/gamepad_offset', self.joy_callback, 10)
        self.sub_grab = self.create_subscription(Bool, '/trigger_grab', self.grab_callback, 10)
        
        # 定时器 20Hz，手感丝滑
        self.timer = self.create_timer(0.05, self.control_loop) 
        self.get_logger().info('=== 手柄+视觉融合控制节点已启动 (Y键抓取, B键张开) ===')

    def vis_callback(self, msg):
        if len(msg.data) < 4: return
        cx, cy, w_px, h_px, conf = msg.data
        if w_px < 15: return
        Z_raw = (self.fx * self.REAL_WIDTH) / w_px
        if Z_raw < 0.08 or Z_raw > 0.60: return
        
        # 更新视觉基准坐标
        self.vis_x = (cx - self.cx_cam) * Z_raw / self.fx
        self.vis_y = (cy - self.cy_cam) * Z_raw / self.fy
        self.vis_z = Z_raw

    def joy_callback(self, msg):
        if len(msg.data) >= 3:
            dx, dy, dz = msg.data[0], msg.data[1], msg.data[2]
            # 累加偏移量
            if abs(dx) > 0: self.off_x += dx * self.speed
            if abs(dy) > 0: self.off_y += dy * self.speed
            if abs(dz) > 0: self.off_z += dz * self.speed
            
            # 限制最大偏移范围 (例如前后左右最多偏离视觉目标 20cm)
            self.off_x = max(-0.20, min(0.20, self.off_x))
            self.off_y = max(-0.20, min(0.20, self.off_y))
            self.off_z = max(-0.15, min(0.15, self.off_z))

    def grab_callback(self, msg):
        if msg.data:
            self.get_logger().info('🎯 Y键触发，夹爪闭合！')
            self.arm.Arm_serial_servo_write(6, 150, 500) 
        else:
            self.get_logger().info('👉 B键触发，夹爪张开！')
            self.arm.Arm_serial_servo_write(6, 30, 500)

    def solve_ik(self, X, Y, Z):
        theta1 = math.atan2(Y, X)
        R = math.sqrt(X**2 + Y**2)
        S = Z - self.base_height
        D = math.sqrt(R**2 + S**2)
        if D > (self.L1 + self.L2 + 0.01) or D < abs(self.L1 - self.L2) - 0.01: return None
        
        cos3 = max(-1.0, min(1.0, (self.L1**2 + self.L2**2 - D**2) / (2 * self.L1 * self.L2)))
        theta3 = math.acos(cos3)
        cos_alpha = max(-1.0, min(1.0, (self.L1**2 + D**2 - self.L2**2) / (2 * self.L1 * D)))
        alpha = math.acos(cos_alpha)
        beta = math.atan2(S, R)
        theta2 = beta - alpha
        theta4 = -(theta2 - (math.pi - theta3))
        
        s1 = max(0.0, min(180.0, 90.0 + math.degrees(theta1)))
        s2 = max(45.0, min(160.0, 180.0 - math.degrees(theta2)))
        s3 = max(50.0, min(125.0, 270.0 - math.degrees(theta3)))
        s5 = max(0.0, min(180.0, 90.0 - math.degrees(theta4)))
        return [s1, s2, s3, s5]

    def control_loop(self):
        # 最终目标 = 视觉绝对坐标 + 摇杆累加偏移量
        final_x = self.vis_x + self.off_x
        final_y = self.vis_y + self.off_y
        final_z = self.vis_z + self.off_z
        
        if abs(final_x) < 0.03: final_x = 0.03 if final_x >= 0 else -0.03
        if abs(final_y) < 0.03: final_y = 0.03 if final_y >= 0 else -0.03
        
        angles = self.solve_ik(final_x, final_y, final_z)
        if angles:
            s1, s2, s3, s5 = angles
            if (abs(s1 - self.last_s1) < 2.0 and abs(s2 - self.last_s2) < 2.0 and abs(s3 - self.last_s3) < 2.0):
                return
            self.arm.Arm_serial_servo_write(1, int(s1), 200)
            self.arm.Arm_serial_servo_write(2, int(s2), 200)
            self.arm.Arm_serial_servo_write(3, int(s3), 200)
            self.arm.Arm_serial_servo_write(4, 90, 200)
            self.arm.Arm_serial_servo_write(5, int(s5), 200)
            self.last_s1, self.last_s2, self.last_s3 = s1, s2, s3

def main(args=None):
    rclpy.init(args=args)
    node = DofbotIKController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
