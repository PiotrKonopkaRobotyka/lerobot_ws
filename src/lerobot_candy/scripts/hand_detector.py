#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import time

class HandDetector(Node):
    def __init__(self):
        super().__init__('hand_detector')
        
        self.bridge = CvBridge()
        self.frame_count = 0
        self.start_time = time.time()
        
        # FOCUSED: Tylko hand detection
        self.window_name = 'Hand Detection'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 700, 500)
        cv2.moveWindow(self.window_name, 800, 50)  # Druga pozycja
        
        # Publishers
        self.hand_pub = self.create_publisher(Point, '/detection/hand_position', 10)
        self.status_pub = self.create_publisher(String, '/hand/status', 10)
        
        # MediaPipe hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Wait for camera
        self.camera_ready = False
        self.check_timer = self.create_timer(1.0, self.check_camera)
        self.image_sub = None
        
        self.get_logger().info('✓ Hand Detector ready - waiting for camera...')
        
    def check_camera(self):
        """Wait for camera to be available"""
        import subprocess
        try:
            result = subprocess.run(['ros2', 'topic', 'list'], capture_output=True, text=True)
            if '/camera/image_raw' in result.stdout and not self.camera_ready:
                self.camera_ready = True
                self.get_logger().info('✓ Camera detected! Starting hand detection...')
                
                self.image_sub = self.create_subscription(
                    Image, '/camera/image_raw', self.detect_hand, 10)
                self.check_timer.cancel()
        except:
            pass
            
    def detect_hand(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = cv2.flip(cv_image, 1)  # Mirror
            
            self.frame_count += 1
            
            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            
            hand_found = False
            if results.multi_hand_landmarks:
                hand_lms = results.multi_hand_landmarks[0]
                lm_list = hand_lms.landmark  # to jest RepeatedCompositeContainer (lista)
                if lm_list and len(lm_list) > 9:
                    lm9 = lm_list[9]  # POJEDYNCZY landmark, ma .x .y .z w [0..1]
                    h, w = frame.shape[:2]
                    palm_x = int(lm9.x * w)
                    palm_y = int(lm9.y * h)
                    # Walidacja zakresów
                    if 0 <= palm_x < w and 0 <= palm_y < h:
                        # Rysuj skeleton + punkt
                        self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                        cv2.circle(frame, (palm_x, palm_y), 10, (255, 0, 0), 2)
                        cv2.putText(frame, 'HAND DETECTED', (max(10, palm_x-80), max(25, palm_y-30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        # Publikuj
                        hand_msg = Point()
                        hand_msg.x = float(palm_x)
                        hand_msg.y = float(palm_y)
                        hand_msg.z = 0.0
                        self.hand_pub.publish(hand_msg)
                        hand_found = True

            # Status + FPS
            self.draw_hand_status(frame, hand_found)
            # Wyświetl
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Hand detection error: {e}')
    
    def draw_hand_status(self, frame, hand_found):
        """Draw hand detection status"""
        h, w = frame.shape[:2]
        
        # Status indicator
        if hand_found:
            cv2.circle(frame, (20, 20), 12, (255, 0, 0), -1)
            cv2.putText(frame, 'HAND FOUND', (40, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.circle(frame, (20, 20), 12, (0, 255, 255), -1)
            cv2.putText(frame, 'SEARCHING HAND', (40, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # FPS
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            fps = self.frame_count / elapsed
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Instructions
        cv2.putText(frame, 'Show your hand to camera', (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def main(args=None):
    rclpy.init(args=args)
    node = HandDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
