#!/usr/bin/env python3                                    
import contextlib                                         
import os                                                 
import cv2                                                
import math                                               
import numpy as np                                        
import time                                               
import sys                                                
import socket                                             # TCP communication
import json                                               # JSON for data serialization
from ultralytics import YOLO                              

class ArmControllerTCP:                                   # Modified class for TCP communication
    def __init__(self, robot_ip="192.168.1.112"):        # Replace with your Pi's IP
        
        # TCP setup
        self.robot_ip = robot_ip                          # Raspberry Pi IP address
        self.robot_port = 8888                            # TCP port for robot communication
        
        # FASTEST model
        self.model = YOLO('yolo11n-pose.pt')             
        
        @contextlib.contextmanager                        
        def suppress_warnings():                          
            with open(os.devnull, 'w') as devnull:       
                old_stderr = os.dup(2)                    
                os.dup2(devnull.fileno(), 2)              
                try:
                    yield                                 
                finally:
                    os.dup2(old_stderr, 2)                
                    os.close(old_stderr)                  

        self.suppress_warnings = suppress_warnings       
        
        # CAMERA SETUP
        self.cap = cv2.VideoCapture(0)                    
        if not self.cap.isOpened():                       
            print("ERROR: Cannot open camera!")          
            sys.exit(1)                                   
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 854)      
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)     
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)         
        
        # Test camera
        ret, test_frame = self.cap.read()                 
        if not ret:                                       
            print("ERROR: Cannot read from camera!")     
            sys.exit(1)                                   
        print(f"Camera OK: {test_frame.shape[1]}x{test_frame.shape[0]}")
        
        # SINGLE ARM ONLY
        self.joint2_angle = 0.0                          
        self.joint3_angle = 0.0                          
        self.has_detection = False                        
        self.limits_ok = True                             
        
        # Robot control
        self.last_robot_command_time = 0                  
        self.robot_command_interval = 0.1                 
        self.robot_commands_sent = 0                      
        self.last_sent_j2 = None                          
        self.last_sent_j3 = None                          
        
        # Minimal tracking
        self.frame_count = 0                              
        self.start_time = time.time()                     
        
        self.arm_mode = 'RIGHT'                           
        self.current_j3_limits = (-145, 145)              
        self.detection_cache_time = 0.5                   
        self.last_good_frame_time = 0                     

        # Test TCP connection
        self.test_robot_connection()

        print("🚀 myCobot Demo TCP", flush=True)      
        print("📹 854x480 for maximum performance", flush=True)     
        print("🔄 Press 'R' for Right Arm, 'L' for Left Arm", flush=True)
        print("📺 Press 'q' to quit", flush=True)

    def test_robot_connection(self):                      # Test TCP connection to robot
        """Test connection to robot"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)                          # 2 second timeout
            sock.connect((self.robot_ip, self.robot_port))
            sock.close()
            print(f"✅ Robot connection OK: {self.robot_ip}:{self.robot_port}")
        except Exception as e:
            print(f"❌ Cannot connect to robot at {self.robot_ip}:{self.robot_port}")
            print(f"   Error: {e}")
            print("   Make sure robot server is running on Pi!")
            sys.exit(1)

    def send_robot_command(self, j2, j3):                 # Send command to robot via TCP
        """Send joint angles to robot via TCP"""
        try:
            # Create command matching your original format
            command = {
                'joint_positions': [-1.570796326795, float(j2), float(j3), 0.0, 1.570796326795, 0.0],
                'timestamp': time.time()
            }
            
            # Send TCP command
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)                          # 1 second timeout
            sock.connect((self.robot_ip, self.robot_port))
            sock.send(json.dumps(command).encode())
            response = sock.recv(1024).decode()
            sock.close()
            
            return response == "OK"
            
        except Exception as e:
            print(f"❌ TCP Error: {e}")
            return False

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
    
        return j2_ok and j3_ok                             

    def run(self):                                        
        """Main loop with TCP communication"""
        window_name = 'myCobot Demo TCP'                  
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE) 
        
        print("Starting main loop with TCP...")           
        
        while True:                                       # Simple loop (no ROS2)
            ret, frame = self.cap.read()                  
            if not ret:                                   
                print("Failed to read frame!")           
                break                                     

            frame = cv2.flip(frame, 1)                    

            self.frame_count += 1                         
            current_time = time.time()                    
            
            # YOLO DETECTION (same as before)
            with self.suppress_warnings():                
                results = self.model(frame, verbose=False, conf=0.1)[0]  
            
            self.last_good_frame_time = current_time      

            # PROCESS ARM (same logic as before)
            self.has_detection = False                    
            
            if results.keypoints is not None and len(results.keypoints.xy) > 0:  
                kpts = results.keypoints.xy[0].cpu().numpy()  
                
                if len(kpts) >= 11:                       
                    if self.arm_mode == 'RIGHT':          
                        shoulder = kpts[5]                
                        elbow = kpts[7]                   
                        wrist = kpts[9]                   
                        arm_color = (0, 255, 0)           
                    else:                                 
                        shoulder = kpts[6]                
                        elbow = kpts[8]                   
                        wrist = kpts[10]                  
                        arm_color = (255, 0, 0)           
                    
                    if not (np.any(np.isnan(shoulder)) or np.any(np.isnan(elbow)) or np.any(np.isnan(wrist)) or  
                            (shoulder[0] < 1 and shoulder[1] < 1) or   
                            (elbow[0] < 1 and elbow[1] < 1) or        
                            (wrist[0] < 1 and wrist[1] < 1)):         
                        
                        vx = elbow[0] - shoulder[0]       
                        vy = elbow[1] - shoulder[1]       

                        j2 = math.atan2(vy, vx) + math.pi/2  
                        j2 = math.atan2(math.sin(j2), math.cos(j2))  
                        
                        ax, ay = elbow[0] - shoulder[0], shoulder[1] - elbow[1]  
                        bx, by = wrist[0] - elbow[0], elbow[1] - wrist[1]        
                        ang1 = math.atan2(ay, ax)         
                        ang2 = math.atan2(by, bx)         
                        j3 = ang1 - ang2                  
                        j3 = math.atan2(math.sin(j3), math.cos(j3))  
                        
                        if self.check_dynamic_limits(j2, j3, self.arm_mode):  

                            self.joint2_angle = j2         
                            self.joint3_angle = j3         
                            self.has_detection = True      
                            self.limits_ok = True          
                            
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
                            s = (int(shoulder[0]), int(shoulder[1]))  
                            e = (int(elbow[0]), int(elbow[1]))        
                            w = (int(wrist[0]), int(wrist[1]))        
                            cv2.line(frame, s, e, (0, 0, 255), 3)    
                            cv2.line(frame, e, w, (0, 0, 255), 3)    
                            cv2.circle(frame, s, 5, (0, 0, 255), -1) 
                            cv2.circle(frame, e, 4, (0, 0, 255), -1) 
                            cv2.circle(frame, w, 4, (0, 0, 255), -1) 

            elif (current_time - self.last_good_frame_time) < self.detection_cache_time:  
                pass                                      
            
            # TCP ROBOT COMMAND (modified from ROS2)
            if (current_time - self.last_robot_command_time) >= self.robot_command_interval:  
                if self.has_detection and self.limits_ok:  
                    j2_deg = math.degrees(self.joint2_angle)  
                    j3_deg = math.degrees(self.joint3_angle)  
                    
                    send_command = True                   
                    if self.last_sent_j2 is not None and self.last_sent_j3 is not None:  
                        if (abs(j2_deg - self.last_sent_j2) < 1.0 and   
                            abs(j3_deg - self.last_sent_j3) < 1.0):     
                            send_command = False          
                    
                    if send_command:                      
                        success = self.send_robot_command(self.joint2_angle, self.joint3_angle)
                        if success:
                            self.robot_commands_sent += 1    
                            self.last_sent_j2 = j2_deg       
                            self.last_sent_j3 = j3_deg       
                            
                            timestamp = time.strftime("%H:%M:%S", time.localtime())  
                            print(f"[{timestamp}] 🤖 {self.arm_mode} #{self.robot_commands_sent:03d}: J2={j2_deg:+4.0f}° J3={j3_deg:+4.0f}°")
                
                self.last_robot_command_time = current_time  
            
            # DISPLAY
            self.draw_simple_status(frame)                
            cv2.imshow(window_name, frame)                
            
            # Handle keys
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
        
        self.cap.release()                                
        cv2.destroyAllWindows()                           

    def draw_simple_status(self, frame):                  # Same as before
        """Simple status display"""
        h, w = frame.shape[:2]                            
        
        if w < 200 or h < 100:                            
            return                                        
        
        mode_color = (0, 255, 0) if self.arm_mode == 'RIGHT' else (255, 255, 0)  
        cv2.putText(frame, f'MODE: {self.arm_mode}', (10, h - 60),   
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 2)
        
        if self.has_detection and self.limits_ok:         
            cv2.circle(frame, (20, 20), 10, (0, 255, 0), -1)       
            cv2.putText(frame, 'ACTIVE', (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  
        elif self.has_detection:                          
            cv2.circle(frame, (20, 20), 10, (0, 0, 255), -1)       
            cv2.putText(frame, 'UNSAFE', (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  
        else:                                             
            cv2.circle(frame, (20, 20), 10, (0, 255, 255), -1)     
            cv2.putText(frame, 'SEARCH', (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  
        
        if self.has_detection:                            
            j2_deg = math.degrees(self.joint2_angle)      
            j3_deg = math.degrees(self.joint3_angle)      
            angle_color = (0, 255, 0) if self.limits_ok else (0, 0, 255)  
            cv2.putText(frame, f'J2:{j2_deg:+4.0f} J3:{j3_deg:+4.0f}',    
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if hasattr(self, 'current_j3_limits'):           
            j3_min, j3_max = self.current_j3_limits      
            limit_color = (255, 255, 0)                  
            cv2.putText(frame, f'J3 Limits: {j3_min:+4.0f} to {j3_max:+4.0f}',  
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, limit_color, 2)

        elapsed = time.time() - self.start_time           
        if elapsed > 0:                                   
            fps = self.frame_count / elapsed              
            cv2.putText(frame, f'FPS:{fps:.1f}', (10, h - 40),      
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, f'Commands:{self.robot_commands_sent}', (10, h - 20),  
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

def main():                                               

    robot_ip = "192.168.1.112"  # Default IP
        
    controller = ArmControllerTCP(robot_ip)               
    
    try:
        controller.run()                                  
    except KeyboardInterrupt:                             
        print("Interrupted")                              

if __name__ == '__main__':                               
    main()                                                