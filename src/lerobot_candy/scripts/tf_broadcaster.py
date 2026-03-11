#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math

class FrameBroadcaster(Node):
    def __init__(self):
        super().__init__('frame_broadcaster')
        
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Publikuj transformacje co 100ms
        self.timer = self.create_timer(0.1, self.broadcast_transforms)
        
        self.get_logger().info('✓ TF2 Frame Broadcaster ready')
    
    def broadcast_transforms(self):
        """Publikuj kluczowe transformacje"""
        current_time = self.get_clock().now().to_msg()
        
        # Transform 1: world -> robot_base
        t_base = TransformStamped()
        t_base.header.stamp = current_time
        t_base.header.frame_id = 'world'
        t_base.child_frame_id = 'robot_base'
        t_base.transform.translation.x = 0.0
        t_base.transform.translation.y = 0.0
        t_base.transform.translation.z = 0.0
        t_base.transform.rotation.w = 1.0
        
        # Transform 2: robot_base -> camera_link
        t_camera = TransformStamped()
        t_camera.header.stamp = current_time
        t_camera.header.frame_id = 'robot_base'
        t_camera.child_frame_id = 'camera_link'
        # DOSTOSUJ te wartości do rzeczywistej pozycji kamery względem robota
        t_camera.transform.translation.x = 0.3   # 30cm przed robotem
        t_camera.transform.translation.y = 0.0   # na środku
        t_camera.transform.translation.z = 0.5   # 50cm nad bazą
        # Kamera skierowana w dół pod kątem 45°
        t_camera.transform.rotation.x = math.sin(math.radians(22.5))
        t_camera.transform.rotation.y = 0.0
        t_camera.transform.rotation.z = 0.0
        t_camera.transform.rotation.w = math.cos(math.radians(22.5))
        
        # Transform 3: robot_base -> table_surface
        t_table = TransformStamped()
        t_table.header.stamp = current_time
        t_table.header.frame_id = 'robot_base'
        t_table.child_frame_id = 'table_surface'
        t_table.transform.translation.x = 0.4   # 40cm przed robotem
        t_table.transform.translation.y = 0.0
        t_table.transform.translation.z = 0.0   # Na poziomie bazy (dostosuj)
        t_table.transform.rotation.w = 1.0
        
        # Publikuj wszystkie transformacje
        self.tf_broadcaster.sendTransform([t_base, t_camera, t_table])

def main(args=None):
    rclpy.init(args=args)
    node = FrameBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()