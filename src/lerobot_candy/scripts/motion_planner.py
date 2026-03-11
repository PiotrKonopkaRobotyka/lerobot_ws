#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Point, PoseStamped, Pose
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
import tf2_ros

# MoveIt2 imports
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, 
    Constraints, 
    JointConstraint,
    WorkspaceParameters,
    PlanningOptions
)
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
import math
import time

class MoveItMotionPlanner(Node):
    def __init__(self):
        super().__init__('moveit_motion_planner')
        
        # TF2 Setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # MoveIt2 Action Client
        self.move_group_action = ActionClient(self, MoveGroup, '/move_action')
        
        # Publishers & Subscribers
        self.status_pub = self.create_publisher(String, '/motion/status', 10)
        self.grasp_sub = self.create_subscription(
            Point, '/detection/candy_position', self.plan_candy_pickup, 10)
        
        # Robot configuration
        self.group_name = "arm"  # Dostosuj do nazwy grupy w MoveIt config
        self.end_effector_link = "tool0"  # Nazwa end-effectora
        self.planning_frame = "robot_base"
        
        # Motion states
        self.current_state = 'idle'
        self.last_joint_state = None
        
        # Predefined poses
        self.home_joints = [0.0, -1.57, 1.57, 0.0, 1.57, 0.0]  # Dostosuj do MyCobot
        self.observe_pose = self.create_pose(0.3, 0.0, 0.4, 0.0, 1.57, 0.0)  # Pozycja obserwacji
        
        # Wait for MoveIt
        self.wait_for_moveit()
        
        self.get_logger().info('✓ MoveIt2 Motion Planner ready')
        self.publish_status("MoveIt2 motion planner initialized")
        
        # Move to observe position at start
        self.move_to_observe_position()
    
    def wait_for_moveit(self):
        """Czekaj na dostępność MoveIt action server"""
        self.get_logger().info('Waiting for MoveIt action server...')
        if not self.move_group_action.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('MoveIt action server not available!')
        else:
            self.get_logger().info('✓ MoveIt action server connected')
    
    def plan_candy_pickup(self, candy_position_msg):
        """Zaplanuj i wykonaj pickup sequence dla wykrytego cukierka"""
        if self.current_state != 'idle':
            self.get_logger().warn('Motion planner busy, ignoring pickup request')
            return
        
        self.current_state = 'planning'
        
        # Konwersja pixel coordinates -> world coordinates
        world_pose = self.pixel_to_world_pose(candy_position_msg)
        
        if world_pose is None:
            self.get_logger().error('Failed to transform candy position to world coordinates')
            self.current_state = 'idle'
            return
        
        self.get_logger().info(
            f'Planning pickup at world coordinates: '
            f'x={world_pose.position.x:.3f}, y={world_pose.position.y:.3f}, z={world_pose.position.z:.3f}'
        )
        
        # Execute pickup sequence
        self.execute_pickup_sequence(world_pose)
    
    def pixel_to_world_pose(self, pixel_point):
        """Konwertuj współrzędne pikseli na współrzędne świata"""
        try:
            # Utwórz pose w ramce kamery
            camera_pose = PoseStamped()
            camera_pose.header.frame_id = 'camera_link'
            camera_pose.header.stamp = self.get_clock().now().to_msg()
            
            # Proste przekształcenie pixel -> camera coordinates
            # UWAGA: Te wartości wymagają kalibracji kamery!
            image_width = 640.0
            image_height = 360.0
            
            # Normalizacja do -1..1
            norm_x = (pixel_point.x - image_width/2) / (image_width/2)
            norm_y = (pixel_point.y - image_height/2) / (image_height/2)
            
            # Przybliżone mapowanie na współrzędne 3D
            # Te wartości trzeba skalibrować!
            camera_pose.pose.position.x = 0.5  # Stała odległość od kamery
            camera_pose.pose.position.y = -norm_x * 0.3  # Lewo-prawo
            camera_pose.pose.position.z = -norm_y * 0.3  # Góra-dół
            
            # Orientacja - chwytaj z góry
            camera_pose.pose.orientation.x = 1.0  # Skierowany w dół
            camera_pose.pose.orientation.y = 0.0
            camera_pose.pose.orientation.z = 0.0
            camera_pose.pose.orientation.w = 0.0
            
            # Transformuj do ramki robota
            transform = self.tf_buffer.lookup_transform(
                self.planning_frame, 'camera_link', rclpy.time.Time())
            
            world_pose = do_transform_pose(camera_pose.pose, transform)
            
            # Dodaj offset nad stołem (bezpieczna wysokość)
            world_pose.position.z = max(world_pose.position.z, 0.05)  # Min 5cm nad stołem
            
            return world_pose
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'TF2 transform error: {e}')
            return None
    
    def execute_pickup_sequence(self, target_pose):
        """Wykonaj sekwencję pickup: pre-grasp -> grasp -> lift -> place"""
        self.current_state = 'executing'
        self.publish_status("Executing pickup sequence")
        
        sequence = [
            ('pre_grasp', self.create_pre_grasp_pose(target_pose)),
            ('grasp', target_pose),
            ('lift', self.create_lift_pose(target_pose)),
            ('place', self.create_place_pose()),
            ('home', None)  # Special case for home position
        ]
        
        for step_name, target in sequence:
            self.get_logger().info(f'Executing step: {step_name}')
            self.publish_status(f"Step: {step_name}")
            
            if step_name == 'home':
                success = self.move_to_joint_target(self.home_joints)
            else:
                success = self.move_to_pose_target(target)
            
            if not success:
                self.get_logger().error(f'Step {step_name} failed')
                self.current_state = 'idle'
                self.publish_status("Pickup sequence failed")
                return
            
            # Wait between steps
            time.sleep(1.0)
        
        self.current_state = 'idle'
        self.publish_status("Pickup sequence completed successfully")
        self.get_logger().info('✓ Pickup sequence completed')
    
    def create_pre_grasp_pose(self, target_pose):
        """Utwórz pozycję pre-grasp (nad celem)"""
        pre_grasp = Pose()
        pre_grasp.position.x = target_pose.position.x
        pre_grasp.position.y = target_pose.position.y
        pre_grasp.position.z = target_pose.position.z + 0.1  # 10cm wyżej
        pre_grasp.orientation = target_pose.orientation
        return pre_grasp
    
    def create_lift_pose(self, target_pose):
        """Utwórz pozycję lift (po podniesieniu)"""
        lift = Pose()
        lift.position.x = target_pose.position.x
        lift.position.y = target_pose.position.y
        lift.position.z = target_pose.position.z + 0.15  # 15cm wyżej
        lift.orientation = target_pose.orientation
        return lift
    
    def create_place_pose(self):
        """Utwórz pozycję place (strefa zrzutu)"""
        place = Pose()
        place.position.x = -0.2  # Po lewej stronie robota
        place.position.y = 0.3
        place.position.z = 0.1
        place.orientation.x = 1.0  # Skierowany w dół
        place.orientation.w = 0.0
        return place
    
    def move_to_pose_target(self, target_pose):
        """Użyj MoveIt do ruchu do zadanej pozycji"""
        try:
            # Utwórz goal dla MoveIt action
            goal_msg = MoveGroup.Goal()
            goal_msg.request.group_name = self.group_name
            goal_msg.request.num_planning_attempts = 5
            goal_msg.request.max_velocity_scaling_factor = 0.5
            goal_msg.request.max_acceleration_scaling_factor = 0.5
            
            # Constraint dla pozycji end-effectora
            pose_goal = PoseStamped()
            pose_goal.header.frame_id = self.planning_frame
            pose_goal.header.stamp = self.get_clock().now().to_msg()
            pose_goal.pose = target_pose
            
            goal_msg.request.goal_constraints = [self.create_pose_constraint(pose_goal)]
            
            # Wyślij goal i czekaj na rezultat
            future = self.move_group_action.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            
            if not future.result():
                self.get_logger().error('MoveIt action goal failed')
                return False
            
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error('MoveIt action goal rejected')
                return False
            
            # Czekaj na zakończenie
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
            
            result = result_future.result()
            if result and result.result.error_code.val == 1:  # SUCCESS
                self.get_logger().info('✓ MoveIt motion completed successfully')
                return True
            else:
                self.get_logger().error(f'MoveIt motion failed with error code: {result.result.error_code.val if result else "timeout"}')
                return False
                
        except Exception as e:
            self.get_logger().error(f'MoveIt motion error: {e}')
            return False
    
    def move_to_joint_target(self, joint_values):
        """Ruch do zadanych wartości stawów"""
        # Podobna implementacja jak move_to_pose_target, ale z joint constraints
        # Implementacja skrócona dla przejrzystości
        self.get_logger().info(f'Moving to joint target: {joint_values}')
        return True  # Uproszczenie
    
    def move_to_observe_position(self):
        """Przenieś robota do pozycji obserwacji"""
        self.get_logger().info('Moving to observe position...')
        success = self.move_to_pose_target(self.observe_pose)
        if success:
            self.publish_status("Ready for candy detection")
        else:
            self.publish_status("Failed to reach observe position")
    
    def create_pose_constraint(self, pose_stamped):
        """Utwórz constraint dla pozycji end-effectora"""
        constraint = Constraints()
        constraint.name = "pose_constraint"
        
        # Tu należy dodać szczegółowe constraints dla pozycji i orientacji
        # Implementacja skrócona dla przejrzystości
        
        return constraint
    
    def create_pose(self, x, y, z, rx, ry, rz):
        """Helper do tworzenia Pose z euler angles"""
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        
        # Konwersja euler angles do quaternion (uproszczona)
        pose.orientation.x = math.sin(rx/2)
        pose.orientation.y = math.sin(ry/2)
        pose.orientation.z = math.sin(rz/2)
        pose.orientation.w = math.cos(rx/2) * math.cos(ry/2) * math.cos(rz/2)
        
        return pose
    
    def publish_status(self, message):
        """Publikuj status ruchu"""
        status_msg = String()
        status_msg.data = message
        self.status_pub.publish(status_msg)
        self.get_logger().info(f'Motion status: {message}')

def main(args=None):
    rclpy.init(args=args)
    node = MoveItMotionPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()