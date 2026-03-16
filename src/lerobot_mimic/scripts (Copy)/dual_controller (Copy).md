#!/usr/bin/env python3
"""
Dual Arm Controller - Computer Vision Robot Control
Controls both Gazebo simulation and real MyCobot robot using hand pose detection
"""

import contextlib                                         # Context managers for suppressing warnings
import os                                                 # Operating system interface
import cv2                                                # OpenCV for computer vision
import math                                               # Mathematical functions
import numpy as np                                        # Numerical arrays and operations
import time                                               # Time functions
import sys                                                # System functions
import socket                                             # TCP communication
import json                                               # JSON for data serialization
from ultralytics import YOLO                              # YOLO pose detection model

# ROS2 imports
import rclpy                                              # ROS2 Python client library
from rclpy.node import Node                               # Base class for ROS2 nodes
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # Robot movement messages
from builtin_interfaces.msg import Duration              # Time duration messages

class DualArmController(Node):
    """
    Dual robot controller that manages both Gazebo simulation and real robot
    Uses computer vision to track hand movements and convert to joint commands
    """
    
    def __init__(self, robot_ip="192.168.1.112", enable_gazebo=True, enable_robot=True):
        """
        Initialize the dual arm controller
        
        Args:
            robot_ip (str): IP address of the real robot
            enable_gazebo (bool): Enable Gazebo simulation output
            enable_robot (bool): Enable real robot TCP output
        """
        super().__init__('dual_arm_controller')           # Initialize ROS2 node
        
        # Home position configuration (in radians: -90°, -135°, 145°, 44°, 90°, 0°)
        self.home_position = [-1.5708, -2.3562, 2.5307, 0.7679, 1.5708, 0.0]
        self.is_going_home = False                        # Flag indicating home movement
        self.home_start_time = 0.0                        # Timestamp when home movement started
        
        # Communication settings
        self.enable_gazebo = enable_gazebo                # Enable/disable Gazebo output
        self.enable_robot = enable_robot                  # Enable/disable real robot output
        self.robot_ip = robot_ip                          # Robot IP address
        self.robot_port = 8888                            # TCP port for robot communication
        
        # ROS2 publisher for Gazebo simulation
        if self.enable_gazebo:
            self.joint_pub = self.create_publisher(       
                JointTrajectory,                          
                '/arm_controller/joint_trajectory',       # Topic for Gazebo joint commands
                10                                        # Queue size
            )
            print("✅ Gazebo ROS2 publisher enabled")
        
        # TCP setup for real robot
        if self.enable_robot:
            self.test_robot_connection()
            print("✅ Real robot TCP enabled")
        
        # Computer vision setup
        self.model = YOLO('yolo11n-pose.pt')             # Load YOLO pose detection model
        
        @contextlib.contextmanager                        
        def suppress_warnings():                          # Context manager to hide YOLO warnings
            with open(os.devnull, 'w') as devnull:       
                old_stderr = os.dup(2)                    
                os.dup2(devnull.fileno(), 2)              
                try:
                    yield                                 
                finally:
                    os.dup2(old_stderr, 2)                
                    os.close(old_stderr)                  

        self.suppress_warnings = suppress_warnings       
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)                    # Open default camera
        if not self.cap.isOpened():                       
            print("ERROR: Cannot open camera!")          
            sys.exit(1)                                   
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)      # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)     
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)         # Minimize buffer for real-time
        
        # Test camera
        ret, test_frame = self.cap.read()                 
        if not ret:                                       
            print("ERROR: Cannot read from camera!")     
            sys.exit(1)                                   
        print(f"Camera OK: {test_frame.shape[1]}x{test_frame.shape[0]}")
        
        # Arm detection variables
        self.joint2_angle = 0.0                          # Current J2 angle (shoulder)
        self.joint3_angle = 0.0                          # Current J3 angle (elbow)
        self.has_detection = False                        # Flag: arm detected in current frame
        self.limits_ok = True                             # Flag: angles within safe limits
        
        # Robot control timing
        self.last_robot_command_time = 0                  # Timestamp of last command sent
        self.robot_command_interval = 1                # Interval between commands (seconds)
        self.gazebo_commands_sent = 0                     # Counter: commands sent to Gazebo
        self.robot_commands_sent = 0                      # Counter: commands sent to robot
        self.last_sent_j2 = None                          # Last J2 angle sent (avoid spam)
        self.last_sent_j3 = None                          # Last J3 angle sent (avoid spam)
        
        # Performance tracking
        self.frame_count = 0                              # Total frames processed
        self.start_time = time.time()                     # Program start time
        
        # Arm detection configuration
        self.arm_mode = 'RIGHT'                           # Current arm mode: 'RIGHT' or 'LEFT'
        self.current_j3_limits = (-145, 145)              # Current J3 limits (dynamic)
        self.detection_cache_time = 0.5                   # How long to cache last detection
        self.last_good_frame_time = 0                     # Time of last successful detection

        # Print startup information
        print("🚀 Dual Control: Gazebo + Real Robot")
        print(f"📡 Gazebo: {'ON' if enable_gazebo else 'OFF'}")
        print(f"🤖 Robot: {'ON' if enable_robot else 'OFF'}")      
        print("📹 854x480 for maximum performance")
        print("🔄 Press 'R' for Right Arm, 'L' for Left Arm")
        print("⚙️  Press 'G' to toggle Gazebo, 'T' to toggle Robot")
        print("🏠 Press 'H' to send robot to HOME position")  # Added missing instruction
        print("📺 Press 'q' to quit")

    def test_robot_connection(self):
        """Test TCP connection to the real robot"""
        if not self.enable_robot:
            return
            
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)                          # 2 second timeout
            sock.connect((self.robot_ip, self.robot_port))
            sock.close()
            print(f"✅ Robot connection OK: {self.robot_ip}:{self.robot_port}")
        except Exception as e:
            print(f"❌ Cannot connect to robot at {self.robot_ip}:{self.robot_port}")
            print(f"   Error: {e}")
            print("   Robot TCP will be disabled!")
            self.enable_robot = False

    def send_to_gazebo(self, j2, j3):
        """
        Send joint command to Gazebo simulation via ROS2
        
        Args:
            j2 (float): Joint 2 angle in radians
            j3 (float): Joint 3 angle in radians
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enable_gazebo:
            return False
            
        try:
            msg = JointTrajectory()                       
            msg.joint_names = ['link1_to_link2', 'link2_to_link3', 'link3_to_link4',  
                             'link4_to_link5', 'link5_to_link6', 'link6_to_link6_flange']
            
            point = JointTrajectoryPoint()                
            # Set joint positions: base(-90°), j2, j3, wrist1(0°), wrist2(90°), wrist3(0°)
            point.positions = [-1.570796326795, float(j2), float(j3), 0.0, 1.570796326795, 0.0]  
            point.time_from_start.sec = 0              
            point.time_from_start.nanosec = 800_000_000  # 0.8 seconds to reach position
            msg.points = [point]                          
            
            self.joint_pub.publish(msg)                   
            self.gazebo_commands_sent += 1
            return True
            
        except Exception as e:
            print(f"❌ Gazebo ROS2 Error: {e}")
            return False

    def send_to_robot(self, j2, j3):
        """
        Send joint command to real robot via TCP
        
        Args:
            j2 (float): Joint 2 angle in radians
            j3 (float): Joint 3 angle in radians
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enable_robot:
            return False
            
        try:
            # Round angles to avoid precision errors
            j2_rounded = round(j2, 6)
            j3_rounded = round(j3, 6)
            
            command = {
                'joint_positions': [-1.570796326795, float(j2_rounded), float(j3_rounded), 
                                  0.0, 1.570796326795, 0.0],
                'timestamp': time.time()
            }
            
            # Send TCP command
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)                          
            sock.connect((self.robot_ip, self.robot_port))
            sock.send(json.dumps(command).encode())
            response = sock.recv(1024).decode()
            sock.close()
            
            if response == "OK":
                self.robot_commands_sent += 1
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ Robot TCP Error: {e}")
            return False

    def send_dual_command(self, j2, j3):
        """
        Send command to both Gazebo and real robot simultaneously
        
        Args:
            j2 (float): Joint 2 angle in radians
            j3 (float): Joint 3 angle in radians
            
        Returns:
            tuple: (success_flag, status_string)
        """
        gazebo_success = False
        robot_success = False
        
        # Send to Gazebo simulation
        if self.enable_gazebo:
            gazebo_success = self.send_to_gazebo(j2, j3)
        
        # Send to real robot
        if self.enable_robot:
            robot_success = self.send_to_robot(j2, j3)
        
        # Create status report
        status = []
        if self.enable_gazebo:
            status.append(f"Gazebo:{'✅' if gazebo_success else '❌'}")
        if self.enable_robot:
            status.append(f"Robot:{'✅' if robot_success else '❌'}")
        
        return gazebo_success or robot_success, " | ".join(status)
    
    def send_to_home_position(self):
        """
        Send both robot and simulation to predefined home position
        Blocks normal commands for 8 seconds (3s move + 5s hold)
        
        Returns:
            bool: True if at least one target was successful
        """
        try:
            home_success = False

            # Send to Gazebo simulation
            if self.enable_gazebo:
                try:
                    msg = JointTrajectory()                       
                    msg.joint_names = ['link1_to_link2', 'link2_to_link3', 'link3_to_link4',  
                                     'link4_to_link5', 'link5_to_link6', 'link6_to_link6_flange']

                    point = JointTrajectoryPoint()                
                    point.positions = self.home_position
                    point.time_from_start.sec = 3         # 3 seconds to reach home
                    point.time_from_start.nanosec = 0     
                    msg.points = [point]                          

                    self.joint_pub.publish(msg)
                    print("🏠 Gazebo: Moving to HOME position...")
                    home_success = True

                except Exception as e:
                    print(f"❌ Gazebo HOME Error: {e}")

            # Send to real robot
            if self.enable_robot:
                try:
                    # Use exact degree values to avoid precision errors
                    home_degrees = [-90.0, -135.0, 145.0, 44.0, 90.0, 0.0]
                    home_radians = [math.radians(deg) for deg in home_degrees]

                    command = {
                        'joint_positions': home_radians,
                        'timestamp': time.time(),
                        'home_command': True  # Special flag for home command
                    }

                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2.0)                          
                    sock.connect((self.robot_ip, self.robot_port))
                    sock.send(json.dumps(command).encode())
                    response = sock.recv(1024).decode()
                    sock.close()

                    if response == "OK":
                        print(f"🏠 Robot: Moving to HOME position: {[f'{d:+4.0f}°' for d in home_degrees]}")
                        home_success = True
                    else:
                        print("❌ Robot: HOME command failed")

                except Exception as e:
                    print(f"❌ Robot HOME Error: {e}")

            # Start blocking period if successful
            if home_success:
                self.is_going_home = True
                self.home_start_time = time.time()  
                print("🏠 Blocking ALL commands for 8 seconds (3s move + 5s hold)...")

            return home_success

        except Exception as e:
            print(f"❌ Critical HOME Error: {e}")
            return False

    def check_dynamic_limits(self, j2, j3, arm_mode):
        """
        Check if joint angles are within dynamic safety limits
        J3 limits change based on J2 position to prevent collisions
        
        Args:
            j2 (float): Joint 2 angle in radians
            j3 (float): Joint 3 angle in radians
            arm_mode (str): Current arm mode ('RIGHT' or 'LEFT')
            
        Returns:
            bool: True if angles are safe, False otherwise
        """
        j2_deg = math.degrees(j2)                         
        j3_deg = math.degrees(j3)                         
    
        # J2 permanent limits: -110° to +110°
        j2_ok = -110 <= j2_deg <= 110                     
    
        # Calculate dynamic J3 limits based on J2 position
        if j2_deg <= 0:  # LEFT SIDE: J2 is negative or zero
            j3_max = 145                                  # J3 max stays constant
            # J3 min varies linearly from 0° (at J2=-110°) to -145° (at J2=0°)
            j3_min = -145 * (j2_deg + 110) / 110
            
        else:  # RIGHT SIDE: J2 is positive
            j3_min = -145                                 # J3 min stays constant
            # J3 max varies linearly from +145° (at J2=0°) to 0° (at J2=+110°)
            j3_max = 145 - 145 * j2_deg / 110
        
        j3_ok = j3_min <= j3_deg <= j3_max
        self.current_j3_limits = (j3_min, j3_max)        # Store for display
    
        # Print limit violations
        if not j2_ok or not j3_ok:                        
            timestamp = time.strftime("%H:%M:%S", time.localtime())  
            print(f"\n[{timestamp}] ⚠️  LIMIT VIOLATION - {arm_mode} ARM:", flush=True)  
    
            if not j2_ok:                                 
                print(f"   J2: {j2_deg:+6.1f}° (limit: -110° to +110°)", flush=True)  
    
            if not j3_ok:                                 
                print(f"   J3: {j3_deg:+6.1f}° (limit: {j3_min:+4.0f}° to {j3_max:+4.0f}°)", flush=True)  
    
            print(f"   At J2={j2_deg:+4.0f}°: J3 range = {j3_min:+4.0f}° to {j3_max:+4.0f}°", flush=True)  
    
        return j2_ok and j3_ok

    def run(self):
        """
        Main control loop
        Processes camera frames, detects hand poses, and sends robot commands
        """
        window_name = 'Dual Control: Gazebo + Robot'                  
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) 
        
        print("Starting dual control loop...")           
        
        while rclpy.ok():                                 
            ret, frame = self.cap.read()                  
            if not ret:                                   
                print("Failed to read frame!")           
                break                                     

            frame = cv2.flip(frame, 1)                    # Mirror camera feed

            self.frame_count += 1                         
            current_time = time.time()                    
            
            # YOLO pose detection
            with self.suppress_warnings():                
                results = self.model(frame, verbose=False, conf=0.1)[0]  
            
            self.last_good_frame_time = current_time      

            # Process arm detection
            self.has_detection = False                    
            
            if results.keypoints is not None and len(results.keypoints.xy) > 0:  
                kpts = results.keypoints.xy[0].cpu().numpy()  # Get keypoints for first person
                
                if len(kpts) >= 11:                       # Need at least 11 keypoints for arms
                    # Select keypoints based on arm mode
                    if self.arm_mode == 'RIGHT':          
                        shoulder = kpts[5]                # Left shoulder (appears right after flip)
                        elbow = kpts[7]                   # Left elbow
                        wrist = kpts[9]                   # Left wrist
                        arm_color = (0, 255, 0)           # Green for right arm
                    else:                                 
                        shoulder = kpts[6]                # Right shoulder (appears left after flip)
                        elbow = kpts[8]                   # Right elbow
                        wrist = kpts[10]                  # Right wrist
                        arm_color = (255, 0, 0)           # Blue for left arm
                    
                    # Validate keypoints
                    if not (np.any(np.isnan(shoulder)) or np.any(np.isnan(elbow)) or np.any(np.isnan(wrist)) or  
                            (shoulder[0] < 1 and shoulder[1] < 1) or   
                            (elbow[0] < 1 and elbow[1] < 1) or        
                            (wrist[0] < 1 and wrist[1] < 1)):         
                        
                        # Calculate joint angles from keypoints
                        vx = elbow[0] - shoulder[0]       # Vector from shoulder to elbow
                        vy = elbow[1] - shoulder[1]       

                        j2 = math.atan2(vy, vx) + math.pi/2        # Shoulder angle
                        j2 = math.atan2(math.sin(j2), math.cos(j2))  # Normalize to [-π, π]
                        
                        # Calculate elbow angle using arm segment vectors
                        ax, ay = elbow[0] - shoulder[0], shoulder[1] - elbow[1]    # Upper arm vector
                        bx, by = wrist[0] - elbow[0], elbow[1] - wrist[1]          # Forearm vector
                        ang1 = math.atan2(ay, ax)         
                        ang2 = math.atan2(by, bx)         
                        j3 = ang1 - ang2                  # Elbow angle
                        j3 = math.atan2(math.sin(j3), math.cos(j3))  # Normalize to [-π, π]
                        
                        # Check safety limits
                        if self.check_dynamic_limits(j2, j3, self.arm_mode):  

                            self.joint2_angle = j2         
                            self.joint3_angle = j3         
                            self.has_detection = True      
                            self.limits_ok = True          
                            
                            # Draw valid arm in green/blue
                            s = (int(shoulder[0]), int(shoulder[1]))  
                            e = (int(elbow[0]), int(elbow[1]))        
                            w = (int(wrist[0]), int(wrist[1]))        
                            cv2.line(frame, s, e, arm_color, 3)      
                            cv2.line(frame, e, w, arm_color, 3)      
                            cv2.circle(frame, s, 5, arm_color, -1)   
                            cv2.circle(frame, e, 4, arm_color, -1)   
                            cv2.circle(frame, w, 4, arm_color, -1)   
                        else:                             
                            self.limits_ok = False         
                            # Draw unsafe arm in red
                            s = (int(shoulder[0]), int(shoulder[1]))  
                            e = (int(elbow[0]), int(elbow[1]))        
                            w = (int(wrist[0]), int(wrist[1]))        
                            cv2.line(frame, s, e, (0, 0, 255), 3)    
                            cv2.line(frame, e, w, (0, 0, 255), 3)    
                            cv2.circle(frame, s, 5, (0, 0, 255), -1) 
                            cv2.circle(frame, e, 4, (0, 0, 255), -1) 
                            cv2.circle(frame, w, 4, (0, 0, 255), -1) 

            elif (current_time - self.last_good_frame_time) < self.detection_cache_time:  
                pass  # Keep last detection state during cache period
            
            # Robot command sending logic
            if (current_time - self.last_robot_command_time) >= self.robot_command_interval:  
                should_send_commands = True
                
                # Check if home movement is in progress
                if self.is_going_home:
                    elapsed_since_home = time.time() - self.home_start_time
                    if elapsed_since_home < 8.0:          # Block for 8 seconds total
                        should_send_commands = False      # Block command sending
                    else:
                        # Resume normal operation after 8 seconds
                        self.is_going_home = False
                        print("🏠 Home position reached, resuming normal operation")
                        should_send_commands = True

                # Send commands if not blocked and detection is valid
                if should_send_commands and self.has_detection and self.limits_ok:  
                    j2_deg = math.degrees(self.joint2_angle)  
                    j3_deg = math.degrees(self.joint3_angle)  
                    
                    send_command = True
                    
                    # Avoid sending duplicate commands (reduce spam)
                    if self.last_sent_j2 is not None and self.last_sent_j3 is not None:  
                        if (abs(j2_deg - self.last_sent_j2) < 1.0 and   
                            abs(j3_deg - self.last_sent_j3) < 1.0):     
                            send_command = False          
                    
                    if send_command:                      
                        success, status = self.send_dual_command(self.joint2_angle, self.joint3_angle)
                        
                        if success:
                            self.last_sent_j2 = j2_deg       
                            self.last_sent_j3 = j3_deg       
                            
                            timestamp = time.strftime("%H:%M:%S", time.localtime())  
                            print(f"[{timestamp}] 🎯 {self.arm_mode} J2={j2_deg:+4.0f}° J3={j3_deg:+4.0f}° | {status}")
                
                self.last_robot_command_time = current_time  
            
            # Display status and video
            self.draw_dual_status(frame)                
            cv2.imshow(window_name, frame)                
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF                   
            if key == ord('q'):                           
                print("Quitting...")                     
                break                                     
            elif key == ord('r') or key == ord('R'):      
                if self.arm_mode != 'RIGHT':              
                    self.arm_mode = 'RIGHT'               
                    print("🔄 Switched to RIGHT ARM mode", flush=True)  
            elif key == ord('l') or key == ord('L'):      
                if self.arm_mode != 'LEFT':               
                    self.arm_mode = 'LEFT'                
                    print("🔄 Switched to LEFT ARM mode", flush=True)
            elif key == ord('g') or key == ord('G'):      
                self.enable_gazebo = not self.enable_gazebo
                print(f"🎮 Gazebo: {'ON' if self.enable_gazebo else 'OFF'}", flush=True)
            elif key == ord('t') or key == ord('T'):      
                self.enable_robot = not self.enable_robot
                print(f"🤖 Robot TCP: {'ON' if self.enable_robot else 'OFF'}", flush=True)
            elif key == ord('h') or key == ord('H'):      
                print("🏠 Sending robot to HOME position...")
                self.send_to_home_position()

            # Process ROS2 callbacks
            rclpy.spin_once(self, timeout_sec=0.001)     
        
        # Cleanup
        self.cap.release()                                
        cv2.destroyAllWindows()                           

    def draw_dual_status(self, frame):
        """
        Draw status information overlay on the video frame
        
        Args:
            frame: OpenCV image frame to draw on
        """
        h, w = frame.shape[:2]                            
        
        if w < 200 or h < 100:                            # Skip if frame too small
            return                                        
        
        # Current arm mode indicator
        mode_color = (0, 255, 0) if self.arm_mode == 'RIGHT' else (255, 255, 0)  
        cv2.putText(frame, f'MODE: {self.arm_mode}', (10, h - 120),   
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 2)
        
        # Gazebo connection status
        gazebo_color = (0, 255, 0) if self.enable_gazebo else (0, 0, 255)
        cv2.putText(frame, f'GAZEBO: {"ON" if self.enable_gazebo else "OFF"}', (10, h - 100),   
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, gazebo_color, 2)
        
        # Robot connection status
        robot_color = (0, 255, 0) if self.enable_robot else (0, 0, 255)
        cv2.putText(frame, f'ROBOT: {"ON" if self.enable_robot else "OFF"}', (10, h - 80),   
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, robot_color, 2)
        
        # Home movement status
        if self.is_going_home:
            elapsed = time.time() - self.home_start_time
            if elapsed < 3.0:
                remaining = 3.0 - elapsed
                cv2.putText(frame, f'MOVING HOME: {remaining:.1f}s', (10, h - 60),   
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)  # Orange
            else:
                remaining = 8.0 - elapsed
                cv2.putText(frame, f'HOLDING HOME: {remaining:.1f}s', (10, h - 60),   
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)  # Yellow
        
        # Detection status indicator
        if self.has_detection and self.limits_ok:         
            cv2.circle(frame, (20, 20), 10, (0, 255, 0), -1)       # Green circle
            cv2.putText(frame, 'ACTIVE', (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  
        elif self.has_detection:                          
            cv2.circle(frame, (20, 20), 10, (0, 0, 255), -1)       # Red circle
            cv2.putText(frame, 'UNSAFE', (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  
        else:                                             
            cv2.circle(frame, (20, 20), 10, (0, 255, 255), -1)     # Yellow circle
            cv2.putText(frame, 'SEARCH', (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  
        
        # Current joint angles
        if self.has_detection:                            
            j2_deg = math.degrees(self.joint2_angle)      
            j3_deg = math.degrees(self.joint3_angle)      
            angle_color = (0, 255, 0) if self.limits_ok else (0, 0, 255)  
            cv2.putText(frame, f'J2:{j2_deg:+4.0f} J3:{j3_deg:+4.0f}',    
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, angle_color, 2)
        
        # Dynamic limits display
        if hasattr(self, 'current_j3_limits'):           
            j3_min, j3_max = self.current_j3_limits      
            limit_color = (255, 255, 0)                  
            cv2.putText(frame, f'J3 Limits: {j3_min:+4.0f} to {j3_max:+4.0f}',  
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, limit_color, 2)

        # Performance metrics
        elapsed = time.time() - self.start_time           
        if elapsed > 0:                                   
            fps = self.frame_count / elapsed              
            cv2.putText(frame, f'FPS:{fps:.1f}', (10, h - 40),      
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Command counters
        cv2.putText(frame, f'Gazebo:{self.gazebo_commands_sent} Robot:{self.robot_commands_sent}', (10, h - 20),  
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

def main():
    """
    Main function - handles user input and starts the controller
    """
    print("🚀 Dual Arm Controller Setup")
    print("Choose communication modes:")
    
    # Get user preferences for communication targets
    gazebo_input = input("Enable Gazebo simulation? (y/n, default=y): ").strip().lower()
    enable_gazebo = gazebo_input != 'n'
    
    robot_input = input("Enable real robot? (y/n, default=y): ").strip().lower()
    enable_robot = robot_input != 'n'
    
    # Get robot IP address
    robot_ip = "192.168.1.112"  # Default IP
    if enable_robot:
        ip_input = input(f"Robot IP address (default={robot_ip}): ").strip()
        if ip_input:
            robot_ip = ip_input
    
    # Initialize ROS2 and start controller
    rclpy.init()                                          
    
    controller = DualArmController(
        robot_ip=robot_ip,
        enable_gazebo=enable_gazebo,
        enable_robot=enable_robot
    )               
    
    try:
        controller.run()                                  
    except KeyboardInterrupt:                             
        print("Interrupted")                              
    finally:
        controller.destroy_node()                         
        rclpy.shutdown()                                  

if __name__ == '__main__':                               
    main()