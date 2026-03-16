#!/usr/bin/env python3                                    
import contextlib                                         # For context managers (suppress warnings)
import os                                                 # Operating system interface
import cv2                                                # OpenCV for computer vision
import math                                               # Mathematical functions (radians, degrees, atan2)
import numpy as np                                        # Numerical arrays and operations
import time                                               # Time functions for timestamps and delays
import sys                                                # System functions (exit)
from ultralytics import YOLO                              # YOLO pose detection model

# ROS2 imports
import rclpy                                              # ROS2 Python client library
from rclpy.node import Node                               # Base class for ROS2 nodes
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # Robot movement messages
from builtin_interfaces.msg import Duration              # Time duration messages

class ArmController(Node):                                # Define main class inheriting from ROS2 Node
    def __init__(self):                                   # Constructor - runs when class is created
        super().__init__('arm_controller')                # Initialize parent ROS2 node with name
        
        # ROS2 publisher
        self.joint_pub = self.create_publisher(           # Create ROS2 publisher object
            JointTrajectory,                              # Message type for robot joint control
            '/arm_controller/joint_trajectory',           # Topic name robot subscribes to
            10                                            # Queue size for messages
        )
        
        # FASTEST model
        self.model = YOLO('yolo11n-pose.pt')             # Load YOLO pose detection model (nano version = fastest)
        
        @contextlib.contextmanager                        # Decorator to create context manager
        def suppress_warnings():                          # Function to hide YOLO warning messages
            with open(os.devnull, 'w') as devnull:       # Open null device (trash bin for output)
                old_stderr = os.dup(2)                    # Save current error output
                os.dup2(devnull.fileno(), 2)              # Redirect errors to null
                try:
                    yield                                 # Execute code block
                finally:
                    os.dup2(old_stderr, 2)                # Restore original error output
                    os.close(old_stderr)                  # Close saved error output

        self.suppress_warnings = suppress_warnings       # Store warning suppressor for later use
        
        # CAMERA SETUP with better error handling
        self.cap = cv2.VideoCapture(0)                    # Open camera (device 0 = default camera)
        if not self.cap.isOpened():                       # Check if camera opened successfully
            print("ERROR: Cannot open camera!")          # Print error message
            sys.exit(1)                                   # Exit program with error code
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)      # Set camera width resolution
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)     # Set camera height resolution
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)         # Minimize camera buffer for real-time
        
        # Test camera
        ret, test_frame = self.cap.read()                 # Try to read one frame from camera
        if not ret:                                       # Check if frame reading failed
            print("ERROR: Cannot read from camera!")     # Print error message
            sys.exit(1)                                   # Exit program with error code
        print(f"Camera OK: {test_frame.shape[1]}x{test_frame.shape[0]}")  # Print actual camera resolution
        
        # SINGLE ARM ONLY
        self.joint2_angle = 0.0                          # Store calculated angle for robot joint 2 (shoulder)
        self.joint3_angle = 0.0                          # Store calculated angle for robot joint 3 (elbow)
        self.has_detection = False                        # Flag: True if arm is detected in current frame
        self.limits_ok = True                             # Flag: True if angles are within safe limits
        
        # Robot control
        self.last_robot_command_time = 0                  # Timestamp of last command sent to robot
        self.robot_command_interval = 0.1                 # Send commands every 0.1 seconds (10 Hz)
        self.robot_commands_sent = 0                      # Counter of total commands sent
        self.last_sent_j2 = None                          # Last J2 angle sent to robot (to avoid spam)
        self.last_sent_j3 = None                          # Last J3 angle sent to robot (to avoid spam)
        
        # Minimal tracking
        self.frame_count = 0                              # Counter of processed frames (for FPS calculation)
        self.start_time = time.time()                     # Program start time (for FPS calculation)
        
        # Add these lines after your existing joint variables:
        self.arm_mode = 'RIGHT'                           # Current arm mode: 'RIGHT' or 'LEFT'

        self.current_j3_limits = (-145, 145)              # Current J3 limits for display (changes dynamically)
        
        self.detection_cache_time = 0.5                   # How long to keep showing detection after it's lost
        self.last_good_frame_time = 0                     # Time of last successful detection

        # Print startup information
        print("🚀 myCobot Demo", flush=True)      # Program name
        print("📹 854x480 for maximum performance", flush=True)     # Camera info
        print("🔄 Press 'R' for Right Arm, 'L' for Left Arm", flush=True)  # Usage instructions
        print("📺 Press 'q' to quit", flush=True)                   # Quit instruction

    def check_dynamic_limits(self, j2, j3, arm_mode):    
        """Dynamic limits with full range at center and linear growth"""
        j2_deg = math.degrees(j2)                         
        j3_deg = math.degrees(j3)                         
    
        # J2 permanent limits
        j2_ok = -110 <= j2_deg <= 110                     
    
        # Calculate J3 limits based on J2 position with linear growth
        if j2_deg <= 0:  # LEFT SIDE: J2 is negative or zero
            # J3 MAX stays constant at +145°
            j3_max = 145
            # J3 MIN varies linearly from 0° (at J2=-110°) to -145° (at J2=0°)
            j3_min = -145 * (j2_deg + 110) / 110
            
        else:  # RIGHT SIDE: J2 is positive
            # J3 MIN stays constant at -145°
            j3_min = -145
            # J3 MAX varies linearly from +145° (at J2=0°) to 0° (at J2=+110°)
            j3_max = 145 - 145 * j2_deg / 110
        
        # Check if J3 is within calculated limits
        j3_ok = j3_min <= j3_deg <= j3_max
        
        # Store current limits for display
        self.current_j3_limits = (j3_min, j3_max)     
    
        if not j2_ok or not j3_ok:                        
            timestamp = time.strftime("%H:%M:%S", time.localtime())  
            print(f"\n[{timestamp}] ⚠️  LIMIT VIOLATION - {arm_mode} ARM:", flush=True)  
    
            if not j2_ok:                                 
                print(f"   J2: {j2_deg:+6.1f}° (limit: -110° to +110°)", flush=True)  
    
            if not j3_ok:                                 
                print(f"   J3: {j3_deg:+6.1f}° (limit: {j3_min:+4.0f}° to {j3_max:+4.0f}°)", flush=True)  
    
            print(f"   At J2={j2_deg:+4.0f}°: J3 range = {j3_min:+4.0f}° to {j3_max:+4.0f}°", flush=True)  
    
        return j2_ok and j3_ok                            # Return True only if both joints are within limits

    def run(self):                                        # Main program loop
        """Main loop with proper window handling"""
        # Create window properly
        window_name = 'myCobot Demo'                      # Define window name for OpenCV
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) # Create OpenCV window with auto-resize
        
        print("Starting main loop...")                    # Inform user that main loop is starting
        
        while rclpy.ok():                                 # Main loop - runs until ROS2 shutdown
            ret, frame = self.cap.read()                  # Read frame from camera
            if not ret:                                   # If frame reading failed
                print("Failed to read frame!")           # Print error message
                break                                     # Exit main loop

            # FIX CAMERA MIRRORING
            frame = cv2.flip(frame, 1)                    # Flip frame horizontally (mirror effect)

            self.frame_count += 1                         # Increment frame counter
            current_time = time.time()                    # Get current timestamp
            
            # YOLO DETECTION
            with self.suppress_warnings():                # Use context manager to hide YOLO warnings
                results = self.model(frame, verbose=False, conf=0.1)[0]  # Run YOLO pose detection (confidence=0.1)
            
            self.last_good_frame_time = current_time      # Update last frame processing time

            # PROCESS RIGHT ARM
            self.has_detection = False                    # Reset detection flag for this frame
            
            if results.keypoints is not None and len(results.keypoints.xy) > 0:  # If YOLO found keypoints
                kpts = results.keypoints.xy[0].cpu().numpy()  # Get keypoints as numpy array (first person)
                
                if len(kpts) >= 11:                       # If enough keypoints detected (need at least 11 for arms)
                    # Select arm based of mode
                    if self.arm_mode == 'RIGHT':          # If in right arm mode
                        # RIGHT ARM: 
                        shoulder = kpts[5]                # Get left shoulder keypoint (appears right after flip)
                        elbow = kpts[7]                   # Get left elbow keypoint (appears right after flip)
                        wrist = kpts[9]                   # Get left wrist keypoint (appears right after flip)
                        arm_color = (0, 255, 0)           # Set drawing color to green for right arm
                    else:                                 # If in left arm mode
                        # LEFT ARM: 
                        shoulder = kpts[6]                # Get right shoulder keypoint (appears left after flip)
                        elbow = kpts[8]                   # Get right elbow keypoint (appears left after flip)
                        wrist = kpts[10]                  # Get right wrist keypoint (appears left after flip)
                        arm_color = (255, 0, 0)           # Set drawing color to blue for left arm
                    
                    # FAST validation (inline)
                    if not (np.any(np.isnan(shoulder)) or np.any(np.isnan(elbow)) or np.any(np.isnan(wrist)) or  # Check for NaN values
                            (shoulder[0] < 1 and shoulder[1] < 1) or   # Check if shoulder coordinates are too small
                            (elbow[0] < 1 and elbow[1] < 1) or        # Check if elbow coordinates are too small
                            (wrist[0] < 1 and wrist[1] < 1)):         # Check if wrist coordinates are too small
                        
                        # INLINE angle calculation
                        vx = elbow[0] - shoulder[0]       # Calculate X vector from shoulder to elbow
                        vy = elbow[1] - shoulder[1]       # Calculate Y vector from shoulder to elbow

                        j2 = math.atan2(vy, vx) + math.pi/2  # Calculate J2 angle using arctangent + offset
                        j2 = math.atan2(math.sin(j2), math.cos(j2))  # Normalize angle to [-π, π] range
                        
                        ax, ay = elbow[0] - shoulder[0], shoulder[1] - elbow[1]  # Upper arm vector
                        bx, by = wrist[0] - elbow[0], elbow[1] - wrist[1]        # Forearm vector
                        ang1 = math.atan2(ay, ax)         # Angle of upper arm
                        ang2 = math.atan2(by, bx)         # Angle of forearm
                        j3 = ang1 - ang2                  # Calculate J3 as difference between arm segments
                        j3 = math.atan2(math.sin(j3), math.cos(j3))  # Normalize J3 angle to [-π, π] range
                        

                        # INLINE limit check
                        if self.check_dynamic_limits(j2, j3, self.arm_mode):  # Check if angles are within safe limits

                            self.joint2_angle = j2         # Store J2 angle for robot command
                            self.joint3_angle = j3         # Store J3 angle for robot command
                            self.has_detection = True      # Set detection flag to true
                            self.limits_ok = True          # Set limits flag to true

                            
                            # FAST drawing - GREEN
                            s = (int(shoulder[0]), int(shoulder[1]))  # Convert shoulder to integer pixel coordinates
                            e = (int(elbow[0]), int(elbow[1]))        # Convert elbow to integer pixel coordinates
                            w = (int(wrist[0]), int(wrist[1]))        # Convert wrist to integer pixel coordinates
                            cv2.line(frame, s, e, arm_color, 3)      # Draw line from shoulder to elbow
                            cv2.line(frame, e, w, arm_color, 3)      # Draw line from elbow to wrist
                            cv2.circle(frame, s, 5, arm_color, -1)   # Draw filled circle at shoulder
                            cv2.circle(frame, e, 4, arm_color, -1)   # Draw filled circle at elbow
                            cv2.circle(frame, w, 4, arm_color, -1)   # Draw filled circle at wrist
                        else:                             # If angles are outside safe limits
                            self.limits_ok = False         # Set limits flag to false
                            # RED drawing for unsafe
                            s = (int(shoulder[0]), int(shoulder[1]))  # Convert coordinates to integers
                            e = (int(elbow[0]), int(elbow[1]))        # Convert coordinates to integers
                            w = (int(wrist[0]), int(wrist[1]))        # Convert coordinates to integers
                            cv2.line(frame, s, e, (0, 0, 255), 3)    # Draw red line from shoulder to elbow
                            cv2.line(frame, e, w, (0, 0, 255), 3)    # Draw red line from elbow to wrist
                            cv2.circle(frame, s, 5, (0, 0, 255), -1) # Draw red filled circle at shoulder
                            cv2.circle(frame, e, 4, (0, 0, 255), -1) # Draw red filled circle at elbow
                            cv2.circle(frame, w, 4, (0, 0, 255), -1) # Draw red filled circle at wrist

            elif (current_time - self.last_good_frame_time) < self.detection_cache_time:  # If within cache time
                # Keep showing last detection status
                pass                                      # Don't reset detection flags (keep last state)
            
            # ROBOT COMMAND
            if (current_time - self.last_robot_command_time) >= self.robot_command_interval:  # If enough time passed since last command
                if self.has_detection and self.limits_ok:  # If arm detected and angles are safe
                    j2_deg = math.degrees(self.joint2_angle)  # Convert J2 to degrees for comparison
                    j3_deg = math.degrees(self.joint3_angle)  # Convert J3 to degrees for comparison
                    
                    send_command = True                   # Assume we should send command
                    if self.last_sent_j2 is not None and self.last_sent_j3 is not None:  # If we sent commands before
                        if (abs(j2_deg - self.last_sent_j2) < 1.0 and   # If J2 change is less than 1 degree
                            abs(j3_deg - self.last_sent_j3) < 1.0):     # If J3 change is less than 1 degree
                            send_command = False          # Don't send command (avoid spam)
                    
                    if send_command:                      # If we should send command
                        self.robot_commands_sent += 1    # Increment command counter
                        self.last_sent_j2 = j2_deg       # Remember last sent J2 angle
                        self.last_sent_j3 = j3_deg       # Remember last sent J3 angle
                        
                        # ROBOT COMMAND
                        msg = JointTrajectory()           # Create new trajectory message
                        msg.joint_names = ['link1_to_link2', 'link2_to_link3', 'link3_to_link4',  # Define joint names
                                         'link4_to_link5', 'link5_to_link6', 'link6_to_link6_flange']
                        
                        point = JointTrajectoryPoint()    # Create trajectory point
                        point.positions = [-1.570796326795, float(self.joint2_angle), float(self.joint3_angle), 0.0, 1.570796326795, 0.0]  # Set joint positions
                        point.time_from_start.sec = 0     # Set execution time seconds
                        point.time_from_start.nanosec = 100_000_000  # Set execution time nanoseconds (0.1 sec)
                        msg.points = [point]              # Add point to trajectory
                        
                        self.joint_pub.publish(msg)       # Send message to robot
                        
                        timestamp = time.strftime("%H:%M:%S", time.localtime())  # Get current time string
                        
                        print(f"[{timestamp}] 🤖 {self.arm_mode} #{self.robot_commands_sent:03d}: J2={j2_deg:+4.0f}° J3={j3_deg:+4.0f}°")  # Print command info
                    
                
                self.last_robot_command_time = current_time  # Update last command time
            
            # DISPLAY with NO POLISH CHARACTERS
            self.draw_simple_status(frame)                # Draw status information on frame
            
            
            # PROPER window display
            cv2.imshow(window_name, frame)                # Show frame in OpenCV window
            
           # Handle keys
            key = cv2.waitKey(1) & 0xFF                   # Wait for key press (1ms timeout)
            if key == ord('q'):                           # If 'q' key pressed
                print("Quitting...")                     # Print quit message
                break                                     # Exit main loop
            elif key == ord('r') or key == ord('R'):      # If 'R' or 'r' key pressed
                if self.arm_mode != 'RIGHT':              # If not already in right mode
                    self.arm_mode = 'RIGHT'               # Switch to right arm mode
                    print("🔄 Switched to RIGHT ARM mode", flush=True)  # Print mode change
            elif key == ord('l') or key == ord('L'):      # If 'L' or 'l' key pressed
                if self.arm_mode != 'LEFT':               # If not already in left mode
                    self.arm_mode = 'LEFT'                # Switch to left arm mode
                    print("🔄 Switched to LEFT ARM mode", flush=True)   # Print mode change
            
            # MINIMAL ROS2
            rclpy.spin_once(self, timeout_sec=0.001)     # Process ROS2 callbacks (very short timeout)
        
        self.cap.release()                                # Release camera resource
        cv2.destroyAllWindows()                           # Close all OpenCV windows

    def draw_simple_status(self, frame):                  # Function to draw status information on screen
        """Simple status display - NO POLISH CHARACTERS"""
        h, w = frame.shape[:2]                            # Get frame height and width
        
        # Make sure frame is big enough for text
        if w < 200 or h < 100:                            # If frame too small for text
            return                                        # Exit function early
        
        # MODE INDICATOR
        mode_color = (0, 255, 0) if self.arm_mode == 'RIGHT' else (255, 255, 0)  # Green for right, cyan for left
        cv2.putText(frame, f'MODE: {self.arm_mode}', (10, h - 60),   # Draw mode text
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 2)
        
        # Status circle
        if self.has_detection and self.limits_ok:         # If arm detected and safe
            cv2.circle(frame, (20, 20), 10, (0, 255, 0), -1)       # Draw green circle
            cv2.putText(frame, 'ACTIVE', (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # Draw "ACTIVE" text
        elif self.has_detection:                          # If arm detected but unsafe
            cv2.circle(frame, (20, 20), 10, (0, 0, 255), -1)       # Draw red circle
            cv2.putText(frame, 'UNSAFE', (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # Draw "UNSAFE" text
        else:                                             # If no arm detected
            cv2.circle(frame, (20, 20), 10, (0, 255, 255), -1)     # Draw yellow circle
            cv2.putText(frame, 'SEARCH', (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # Draw "SEARCH" text
        
        # Angles - bigger font
        if self.has_detection:                            # If arm is detected
            j2_deg = math.degrees(self.joint2_angle)      # Convert J2 to degrees
            j3_deg = math.degrees(self.joint3_angle)      # Convert J3 to degrees
            angle_color = (0, 255, 0) if self.limits_ok else (0, 0, 255)  # Green if safe, red if unsafe
            cv2.putText(frame, f'J2:{j2_deg:+4.0f} J3:{j3_deg:+4.0f}',    # Draw angle values
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
            # DYNAMIC LIMITS DISPLAY (NEW FEATURE)
        if hasattr(self, 'current_j3_limits'):           # If limits are available
            j3_min, j3_max = self.current_j3_limits      # Get current J3 limits
            limit_color = (255, 255, 0)                  # Yellow color for limits
            cv2.putText(frame, f'J3 Limits: {j3_min:+4.0f} to {j3_max:+4.0f}',  # Draw limit values
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, limit_color, 2)

        elapsed = time.time() - self.start_time           # Calculate elapsed time since start
        if elapsed > 0:                                   # If time has passed
            fps = self.frame_count / elapsed              # Calculate frames per second
            cv2.putText(frame, f'FPS:{fps:.1f}', (10, h - 40),      # Draw FPS counter
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Commands sent
        cv2.putText(frame, f'Commands:{self.robot_commands_sent}', (10, h - 20),  # Draw command counter
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
def main():                                               # Main function - program entry point
    rclpy.init()                                          # Initialize ROS2 system
    controller = ArmController()                           # Create controller object
    
    try:
        controller.run()                                  # Start main program loop
    except KeyboardInterrupt:                             # If Ctrl+C pressed
        print("Interrupted")                              # Print interruption message
    finally:
        controller.destroy_node()                         # Clean up ROS2 node
        rclpy.shutdown()                                  # Shutdown ROS2 system

if __name__ == '__main__':                               # If script run directly (not imported)
    main()                                                # Call main function