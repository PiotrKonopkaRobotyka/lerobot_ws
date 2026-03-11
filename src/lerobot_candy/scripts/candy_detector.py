#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class CandyDetector(Node):
    def __init__(self):
        super().__init__('candy_detector')
        
        self.bridge = CvBridge()
        self.frame_count = 0
        self.start_time = time.time()
        
        # Window setup
        self.window_name = 'Rectangle Detection (Shape-Based)'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 700, 500)
        cv2.moveWindow(self.window_name, 50, 50)
        
        # Publishers
        self.candy_pub = self.create_publisher(Point, '/detection/candy_position', 10)
        self.status_pub = self.create_publisher(String, '/candy/status', 10)
        
        # SHAPE-BASED DETECTION PARAMETERS - ZŁAGODZONE
        self.min_area = 400                    # Zmniejszone z 800
        self.max_area = 15000                  # Zwiększone z 8000
        self.min_aspect_ratio = 0.3            # Zmniejszone z 0.6
        self.max_aspect_ratio = 3.0            # Zwiększone z 1.8
        self.min_solidity = 0.5                # Zmniejszone z 0.75
        self.epsilon_factor = 0.04             # Zwiększone z 0.015
        
        # Edge detection parameters
        self.canny_low = 50
        self.canny_high = 150
        self.blur_kernel = (5, 5)
        
        # Center stabilization
        self.center_history = []
        self.center_avg_len = 5
        self.last_valid_center = None
        
        # Camera waiting
        self.camera_ready = False
        self.check_timer = self.create_timer(1.0, self.check_camera)
        self.image_sub = None
        
        self.get_logger().info('✓ Shape-Based Candy Detector ready - detecting rectangles')
        
    def check_camera(self):
        """Wait for camera to be available"""
        import subprocess
        try:
            result = subprocess.run(['ros2', 'topic', 'list'], capture_output=True, text=True)
            if '/camera/image_raw' in result.stdout and not self.camera_ready:
                self.camera_ready = True
                self.get_logger().info('✓ Camera detected! Starting rectangle detection...')
                
                self.image_sub = self.create_subscription(
                    Image, '/camera/image_raw', self.detect_rectangles, 10)
                self.check_timer.cancel()
        except:
            pass
    
    def detect_rectangles(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame = cv2.flip(cv_image, 1)  # Mirror
            
            self.frame_count += 1
            
            # KROK 1: Preprocessing - edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
            edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
            
            # KROK 2: Znajdź kontury
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # KROK 3: Znajdź najlepszy prostokąt
            best_rectangle = self.find_best_rectangle(contours)
            
            rectangle_found = False
            if best_rectangle:
                rectangle_found = True
                
                # KROK 4: Oblicz precyzyjny środek
                center = self.calculate_precise_center(best_rectangle['contour'])
                
                if center:
                    # KROK 5: Stabilizacja środka
                    stable_center = self.stabilize_center(center)
                    
                    # KROK 6: Rysowanie detekcji
                    self.draw_rectangle_detection(frame, best_rectangle, center, stable_center)
                    
                    # KROK 7: Publikuj pozycję
                    candy_msg = Point()
                    candy_msg.x = float(stable_center[0])
                    candy_msg.y = float(stable_center[1])
                    candy_msg.z = float(best_rectangle['score'])
                    self.candy_pub.publish(candy_msg)
                    
                    self.get_logger().info(
                        f'Rectangle detected: center=({stable_center},{stable_center[1]}), '
                        f'area={best_rectangle["area"]}, ratio={best_rectangle["aspect_ratio"]:.2f}, '
                        f'score={best_rectangle["score"]:.2f}'
                    )
            
            # KROK 8: Status overlay
            self.draw_detection_status(frame, rectangle_found, best_rectangle, edges)
            
            # Display
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Rectangle detection error: {e}')
    
    def find_best_rectangle(self, contours):
        """Znajdź najlepszy prostokąt z ZŁAGODZONYMI parametrami + debug"""
        best_rectangle = None
        best_score = 0
        debug_info = []

        for i, contour in enumerate(contours):
            debug_entry = f"Contour {i}: "

            # Filtr 1: Rozmiar
            area = cv2.contourArea(contour)
            debug_entry += f"area={area:.0f} "

            if area < self.min_area:
                debug_entry += "REJECTED: area too small"
                debug_info.append(debug_entry)
                continue
            if area > self.max_area:
                debug_entry += "REJECTED: area too large"
                debug_info.append(debug_entry)
                continue
            
            # Filtr 2: Aproksymacja do prostokąta
            epsilon = self.epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            debug_entry += f"vertices={len(approx)} "

            # ZŁAGODZONE: Akceptuj 3-6 wierzchołków zamiast dokładnie 4
            if len(approx) < 3 or len(approx) > 6:
                debug_entry += "REJECTED: wrong vertex count"
                debug_info.append(debug_entry)
                continue
            
            # Filtr 3: Aspect ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            debug_entry += f"ratio={aspect_ratio:.2f} "

            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                debug_entry += "REJECTED: aspect ratio"
                debug_info.append(debug_entry)
                continue
            
            # Filtr 4: Solidity (zwartość kształtu)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            debug_entry += f"solidity={solidity:.2f} "

            if solidity < self.min_solidity:
                debug_entry += "REJECTED: solidity too low"
                debug_info.append(debug_entry)
                continue
            
            # Oblicz wynik jakości
            area_score = min(1.0, area / 4000.0)
            aspect_score = 1.0 - abs(aspect_ratio - 1.0)  # Preferuj kwadraty
            solidity_score = solidity
            vertex_score = 1.0 if len(approx) == 4 else 0.8  # Bonus za 4 wierzchołki

            total_score = area_score * 0.25 + aspect_score * 0.25 + solidity_score * 0.25 + vertex_score * 0.25
            debug_entry += f"score={total_score:.2f} "

            if total_score > best_score:
                best_score = total_score
                best_rectangle = {
                    'contour': approx,
                    'area': area,
                    'bbox': (x, y, w, h),
                    'aspect_ratio': aspect_ratio,
                    'solidity': solidity,
                    'score': total_score,
                    'vertices': len(approx)
                }
                debug_entry += "BEST CANDIDATE"

            debug_info.append(debug_entry)

        # Debug output (pierwszych 5 konturów)
        if len(debug_info) > 0:
            print(f"\n=== DEBUG: Found {len(contours)} contours ===")
            for info in debug_info[:5]:
                print(info)
            if best_rectangle:
                print(f"SELECTED: area={best_rectangle['area']}, vertices={best_rectangle['vertices']}, score={best_rectangle['score']:.2f}")
            else:
                print("NO RECTANGLE SELECTED")
            print("=" * 40)

        return best_rectangle
    
    def calculate_precise_center(self, contour):
        """Oblicz precyzyjny środek konturu"""
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
        return None
    
    def stabilize_center(self, center):
        """Stabilizuj środek przez uśrednianie"""
        self.center_history.append(center)
        if len(self.center_history) > self.center_avg_len:
            self.center_history.pop(0)
        
        avg_x = int(sum([c[0] for c in self.center_history]) / len(self.center_history))
        avg_y = int(sum([c[1] for c in self.center_history]) / len(self.center_history))
        
        self.last_valid_center = (avg_x, avg_y)
        return (avg_x, avg_y)
    
    def draw_rectangle_detection(self, frame, rectangle_data, raw_center, stable_center):
        """Rysuj wykryty prostokąt"""
        contour = rectangle_data['contour']
        
        # Narysuj kontur prostokąta (zielony)
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
        
        # Narysuj bounding box (niebieski)
        x, y, w, h = rectangle_data['bbox']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Narysuj środki
        raw_x, raw_y = raw_center
        stable_x, stable_y = stable_center
        
        # Surowy środek (czerwony)
        cv2.circle(frame, (raw_x, raw_y), 5, (0, 0, 255), -1)
        
        # Stabilny środek (fioletowy) - wysyłany do robota
        cv2.circle(frame, (stable_x, stable_y), 8, (255, 0, 255), -1)
        
        # Etykiety
        cv2.putText(frame, 'RECTANGLE', (stable_x-50, stable_y-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'Score: {rectangle_data["score"]:.2f}', (stable_x-40, stable_y+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f'Ratio: {rectangle_data["aspect_ratio"]:.2f}', (stable_x-40, stable_y+35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_detection_status(self, frame, rectangle_found, rectangle_data, edges):
        """Rysuj status detekcji + debug info"""
        h, w = frame.shape[:2]
        
        # Status indicator
        if rectangle_found:
            cv2.circle(frame, (20, 20), 12, (0, 255, 0), -1)
            cv2.putText(frame, 'RECTANGLE FOUND', (40, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Pokaż szczegóły wykrytego prostokąta
            if rectangle_data:
                cv2.putText(frame, f"Vertices: {rectangle_data.get('vertices', 'N/A')}", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Area: {rectangle_data['area']:.0f}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Ratio: {rectangle_data['aspect_ratio']:.2f}", (10, 170), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Solidity: {rectangle_data['solidity']:.2f}", (10, 190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.circle(frame, (20, 20), 12, (0, 255, 255), -1)
            cv2.putText(frame, 'SEARCHING RECTANGLE', (40, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Parametry detekcji
        cv2.putText(frame, f'Area: {self.min_area}-{self.max_area} px', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f'Ratio: {self.min_aspect_ratio:.1f}-{self.max_aspect_ratio:.1f}', (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f'Solidity: >{self.min_solidity:.2f}', (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, f'Vertices: 3-6', (10, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # FPS
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            fps = self.frame_count / elapsed
            cv2.putText(frame, f'FPS: {fps:.1f}', (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Instructions
        cv2.putText(frame, 'Try any rectangular/square object', (10, h - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Debug: Pokaż edges w prawym górnym rogu
        small_edges = cv2.resize(edges, (160, 120))
        small_edges_bgr = cv2.cvtColor(small_edges, cv2.COLOR_GRAY2BGR)
        frame[10:130, w-170:w-10] = small_edges_bgr
        cv2.rectangle(frame, (w-170, 10), (w-10, 130), (255, 255, 255), 1)
        cv2.putText(frame, 'EDGES', (w-160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def main(args=None):
    rclpy.init(args=args)
    node = CandyDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
