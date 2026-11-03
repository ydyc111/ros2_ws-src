#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
import evdev
from evdev import ecodes
import threading

class GamepadNode(Node):
    def __init__(self):
        super().__init__('gamepad_node')
        # 根据你的截图，设备路径已经是 event5
        self.device_path = "/dev/input/event5" 
        
        self.pub_offset = self.create_publisher(Float32MultiArray, '/gamepad_offset', 10)
        self.pub_grab = self.create_publisher(Bool, '/trigger_grab', 10)
        
        self.dx, self.dy, self.dz = 0.0, 0.0, 0.0
        self.deadzone = 0.15
        
        self.get_logger().info(f'尝试连接手柄: {self.device_path}')
        try:
            self.device = evdev.InputDevice(self.device_path)
            self.get_logger().info(f'手柄已连接: {self.device.name}')
            self.thread = threading.Thread(target=self.read_gamepad, daemon=True)
            self.thread.start()
        except Exception as e:
            self.get_logger().error(f'手柄连接失败: {e}')

    def read_gamepad(self):
        try:
            for event in self.device.read_loop():
                if not rclpy.ok(): break
                
                # 摇杆数据
                if event.type == ecodes.EV_ABS:
                    if event.code == ecodes.ABS_X:
                        val = (event.value - 127.5) / 127.5
                        self.dx = val if abs(val) > self.deadzone else 0.0
                    elif event.code == ecodes.ABS_Y:
                        val = (event.value - 127.5) / 127.5
                        self.dy = val if abs(val) > self.deadzone else 0.0
                    elif event.code == ecodes.ABS_RZ:
                        val = (event.value - 127.5) / 127.5
                        self.dz = val if abs(val) > self.deadzone else 0.0
                    
                    msg = Float32MultiArray()
                    msg.data = [self.dx, self.dy, self.dz]
                    self.pub_offset.publish(msg)
                
                # 按键数据 (防抖：只在按下的瞬间触发)
                elif event.type == ecodes.EV_KEY:
                    if event.value == 1: # 1 代表按下瞬间，0 是松开，2 是连发
                        self.get_logger().info(f'🎮 按下了按键: {event.code}')
                        if event.code == ecodes.BTN_THUMB2: # Y 键
                            self.pub_grab.publish(Bool(data=True))
                        elif event.code == ecodes.BTN_THUMB: # B 键
                            self.pub_grab.publish(Bool(data=False))
        except Exception as e:
            self.get_logger().error(f'读取中断: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = GamepadNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
