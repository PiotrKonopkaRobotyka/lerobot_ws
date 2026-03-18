#!/usr/bin/env python3
"""
SO-101 Pose Mimic Node 

"""

import contextlib # do cichego łapania błędów przy ładowaniu modelu YOLO (który jest głośny)
import os # do przekierowania stderr do /dev/null
import cv2 # OpenCV do obsługi kamery i rysowania OSD
import math # do obliczeń kątów i konwersji radianów/stopni
import numpy as np # do operacji na tablicach (np. sprawdzanie NaN)
import time # do pomiaru czasu i FPS
import sys # do obsługi błędów krytycznych (np. brak kamery)

from ultralytics import YOLO # do detekcji kluczowych punktów ciała

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# ===========================================================================
# ⚙️  CALIBRATION CONSTANTS — zmień te wartości jeśli ruch jest odwrócony
# ===========================================================================
J2_SIGN   = -1      # Zmień na +1 jeśli shoulder idzie w złym kierunku
J3_SIGN   = -1      # Zmień na +1 jeśli elbow idzie w złym kierunku
J2_OFFSET = -1.5708      # Dodatkowy offset J2 [rad]
J3_OFFSET = -1.5708      # Dodatkowy offset J3 [rad]

# Home position — bezpieczna poza startowa [rad]
HOME_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0]  # [J1, J2, J3, J4, J5]

# ===========================================================================
# ⚙️  ROBOT CONFIG
# ===========================================================================
JOINT_NAMES    = ["1", "2", "3", "4", "5"]
YOLO_MODEL     = "yolo26m-pose.pt"
CONF_THRESHOLD = 0.3           # pewność detekcji (0-1)
MIN_ANGLE_DEG  = 2.0           # deadband — nie wysyłaj jeśli zmiana jest mniejsza niż 3°      
CONTROL_HZ     = 10             # 2 komend/sek do robota
TRAJ_TIME_S    = 0.08           # robot ma 0.1s na wykonanie ruchu
EMA_ALPHA      = 0.5          # wygładzanie (0.1=wolno, 0.5=szybko)

# SO-101 JOINT LIMITS [rad] — z URDF / ECE4560
J1_LIM = (-1.919, 1.919)  # shoulder_pan / ≈ ±110°
J2_LIM = (-1.74,  1.74)   # shoulder_lift / # ≈ ±100°
J3_LIM = (-1.5708,  1.5708)   # elbow_flex / # ≈ ±90°
J4_LIM = (-1.65,  1.65)   # wrist_flex / # ≈ ±94°
J5_LIM = (-2.74,  2.84)   # wrist_roll / # ≈ ±157° (nie jest używany w mimic)

J2_LIM_DEG = (math.degrees(J2_LIM[0]), math.degrees(J2_LIM[1]))
J3_LIM_DEG = (math.degrees(J3_LIM[0]), math.degrees(J3_LIM[1]))  

# Fixed joints (nie sterowane przez CV)
FIXED_J1 = 0.0 
FIXED_J4 = 0.0
FIXED_J5 = 0.0

# YOLO COCO keypoint indices
KP_L_SHOULDER  = 5 #używamy do detekcji prawego ramienia, bo obraz jest lustrzany
#KP_R_SHOULDER = 6
KP_L_ELBOW = 7  
#KP_R_ELBOW = 8
KP_L_WRIST = 9
#KP_R_WRIST = 10

CAM_W, CAM_H = 854, 480 # rozdzielczość kamery

@contextlib.contextmanager
def _suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old, 2)
            os.close(old)


class SO101MimicNode(Node):

    def __init__(self):
        super().__init__("so101_mimic") # nazwa node'a w ROS2

        self._pub = self.create_publisher(
            JointTrajectory, "/arm_controller/joint_trajectory", 10) # ← TOPIC do robota

        self.get_logger().info(f"⏳ Loading {YOLO_MODEL}...")
        with _suppress_stderr():
            self._model = YOLO(YOLO_MODEL)
        self.get_logger().info("✓ YOLO loaded")

        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self.get_logger().fatal("❌ Cannot open camera!")
            sys.exit(1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, tf = self._cap.read()
        if not ret:
            self.get_logger().fatal("❌ Cannot read camera!")
            sys.exit(1)
        self.get_logger().info(f"✓ Camera {tf.shape[1]}x{tf.shape[0]}")

        # State
        self._arm_mode       = "RIGHT"
        self._j2_smooth      = 0.0 # wygładzona wartość J2 (EMA) 
        self._j3_smooth      = 0.0 # wygładzona wartość J3 (EMA) 
        self._has_detection  = False
        self._limits_ok      = True
        self._last_good_t    = 0.0
        self._elbow_conf     = 0.0
        self._current_j3_lim = J3_LIM_DEG

        # Rate limiting
        self._cmd_interval  = 1.0 / CONTROL_HZ
        self._last_cmd_t    = 0.0
        self._last_j2_sent  = None
        self._last_j3_sent  = None
        self._cmds_sent     = 0

        # Perf
        self._frame_count = 0
        self._start_t     = time.time()
        # self.  → bo potrzebne w innych metodach (draw_osd, run)
        # _      → bo to wewnętrzny detal klasy, nie API

        self._print_help()

    # =======================================================================
    def _print_help(self):
        lines = [
            "═" * 55,
            "SO-101 Pose Mimic — CALIBRATED",
            f"  Model   : {YOLO_MODEL}",
            f"  Conf    : {CONF_THRESHOLD}",
            f"  Deadband: {MIN_ANGLE_DEG}°",
            f"  TrajTime: {TRAJ_TIME_S}s",
            f"  J2_SIGN : {J2_SIGN}   J3_SIGN: {J3_SIGN}",
            "─" * 55,
            "  H     — wyślij do HOME position",
            "  Q     — wyjście",
        ]
        for l in lines:
            self.get_logger().info(l)

    # =======================================================================
    # Safety limits — SO-101 specific
    # =======================================================================
    def _check_limits(self, j2: float, j3: float) -> bool:
        j2d = math.degrees(j2) # konwersja na stopnie dla czytelności logów
        j3d = math.degrees(j3)

        j2_ok = J2_LIM_DEG[0] <= j2d <= J2_LIM_DEG[1] # sprawdzamy limity dla J2
        j3_ok = J3_LIM_DEG[0] <= j3d <= J3_LIM_DEG[1] # sprawdzamy limity dla J3

        if not (j2_ok and j3_ok): # jeśli któryś z kątów jest poza limitem, logujemy ostrzeżenie
            self.get_logger().warn(
                f"⚠️ LIMIT  J2={j2d:+.1f}° {J2_LIM_DEG}  J3={j3d:+.1f}° {J3_LIM_DEG}"
            )
        return j2_ok and j3_ok


    # =======================================================================
    # EMA filter
    # =======================================================================
    def _ema(self, j2_new: float, j3_new: float):
        self._j2_smooth = EMA_ALPHA * j2_new + (1 - EMA_ALPHA) * self._j2_smooth # aktualizacja wygładzonych wartości J2 i J3 za pomocą formuły EMA
        self._j3_smooth = EMA_ALPHA * j3_new + (1 - EMA_ALPHA) * self._j3_smooth

    # =======================================================================
    # Angle computation — poprawiona konwencja dla SO-101
    # =======================================================================
    @staticmethod
    def _compute_angles_raw(shoulder, elbow, wrist):
        """
        Oblicz surowe kąty J2 i J3 na podstawie pozycji barku, łokcia i nadgarstka.
        """
        vx = elbow[0] - shoulder[0]
        vy = elbow[1] - shoulder[1]

        j2 = math.atan2(vy, vx)          # kąt ramienia od poziomej
        j2 = math.atan2(math.sin(j2), math.cos(j2)) # Normalize angle to [-π, π] range

        # Elbow angle: kąt między wektorem bark→łokieć a łokieć→nadgarstek
        ax, ay =  (elbow[0] - shoulder[0]),  (shoulder[1] - elbow[1])
        bx, by =  (wrist[0]  - elbow[0]),    (elbow[1]    - wrist[1])
        ang1 = math.atan2(ay, ax)
        ang2 = math.atan2(by, bx)
        j3 = ang1 - ang2
        j3 = math.atan2(math.sin(j3), math.cos(j3)) # Normalize angle to [-π, π] range

        return j2, j3

    # =======================================================================
    # Apply sign correction + offset
    # =======================================================================
    @staticmethod
    def _apply_calibration(j2_raw: float, j3_raw: float):
        j2 = J2_SIGN * j2_raw + J2_OFFSET   # stałe z góry pliku
        j3 = J3_SIGN * j3_raw + J3_OFFSET
        j2 = math.atan2(math.sin(j2), math.cos(j2)) # Normalize angle to [-π, π] range
        j3 = math.atan2(math.sin(j3), math.cos(j3)) # Normalize angle to [-π, π] range
        return j2, j3

    # =======================================================================
    # Publish trajectory command
    # =======================================================================
    def _publish_cmd(self, j2: float, j3: float):
        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = [FIXED_J1, float(j2), float(j3), FIXED_J4, FIXED_J5]
        pt.time_from_start.sec    = 0
        pt.time_from_start.nanosec = int(TRAJ_TIME_S * 1e9)
        msg.points = [pt]
        self._pub.publish(msg)
        self._cmds_sent += 1

    # =======================================================================
    # Send robot to HOME position
    # =======================================================================
    def _send_home(self):
        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = list(HOME_POSITION)
        pt.time_from_start.sec    = 2   # 2 sekundy do home — powoli!
        pt.time_from_start.nanosec = 0
        msg.points = [pt]
        self._pub.publish(msg)
        h = [f"{math.degrees(a):.1f}°" for a in HOME_POSITION]
        self.get_logger().info(f"🏠 HOME → J1={h[0]} J2={h[1]} J3={h[2]} J4={h[3]} J5={h[4]}")

    # =======================================================================
    # OSD overlay
    # =======================================================================
    def _draw_osd(self, frame):
        h, w = frame.shape[:2]
        if w < 200 or h < 100:
            return

        # Status badge
        if self._has_detection and self._limits_ok:
            bc, label = (0, 200, 0), "ACTIVE"
        elif self._has_detection:
            bc, label = (0, 0, 255), "UNSAFE"
        else:
            bc, label = (0, 200, 200), "SEARCH"

        cv2.circle(frame, (20, 20), 10, bc, -1)
        cv2.putText(frame, label, (35, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        y = 55
        if self._has_detection:
            j2d = math.degrees(self._j2_smooth)
            j3d = math.degrees(self._j3_smooth)
            col = (0, 200, 0) if self._limits_ok else (0, 0, 255)
            cv2.putText(frame, f"J2={j2d:+5.1f}  J3={j3d:+5.1f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2); y += 25

            q = int(self._elbow_conf * 100)    # jakość detekcji łokcia w procentach
            qc = (0, 200, 0) if q >= 70 else (0, 140, 255) # zielony powyżej 70%, pomarańczowy poniżej
            cv2.putText(frame, f"Quality {q}%",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, qc, 2); y += 22

        cv2.putText(frame, f"J3 lim [{J3_LIM_DEG[0]:.0f},{J3_LIM_DEG[1]:.0f}]",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1); y += 20
        # Bottom bar
        mc = (0, 220, 0) if self._arm_mode == "RIGHT" else (0, 220, 220)
        cv2.putText(frame, f"MODE {self._arm_mode}", (10, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mc, 2)

        elapsed = time.time() - self._start_t
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS {fps:.1f}", (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

        cv2.putText(frame, f"Cmds {self._cmds_sent}", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 2)

        cv2.putText(frame, f"SIGN J2={J2_SIGN:+d} J3={J3_SIGN:+d}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # =======================================================================
    # Main loop
    # =======================================================================
    def run(self):
        win = "SO-101 Mimic"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

        # Wyślij do HOME na starcie
        time.sleep(0.5)
        self._send_home()

        self.get_logger().info("🚀 Main loop started")

        while rclpy.ok():
            ret, frame = self._cap.read()
            if not ret:
                self.get_logger().error("❌ Camera read failed!")
                break

            frame = cv2.flip(frame, 1)
            self._frame_count += 1
            now = time.time()

            # YOLO
            results = self._model(frame, verbose=False, conf=0.10)
            self._has_detection = False

            if (results[0].keypoints is not None
                    and len(results[0].keypoints.xy) > 0):
                kpts = results[0].keypoints.xy[0].cpu().numpy()
                conf = (results[0].keypoints.conf[0].cpu().numpy()
                        if results[0].keypoints.conf is not None else None)

                if len(kpts) > 10:
                    if self._arm_mode == "RIGHT":
                        si, ei, wi = KP_L_SHOULDER, KP_L_ELBOW, KP_L_WRIST
                        arm_color  = (0, 220, 0)
                    #else:
                    #    si, ei, wi = KP_R_SHOULDER, KP_R_ELBOW, KP_R_WRIST
                    #    arm_color  = (0, 220, 220)

                    shoulder, elbow, wrist = kpts[si], kpts[ei], kpts[wi]
                    self._elbow_conf = conf[ei] if conf is not None else 1.0

                    valid = (
                        self._elbow_conf >= CONF_THRESHOLD
                        and all(p[0] > 1 and p[1] > 1 for p in (shoulder, elbow, wrist))
                        and not any(np.isnan(p).any() for p in (shoulder, elbow, wrist))
                    )

                    if valid:
                        j2r, j3r = self._compute_angles_raw(shoulder, elbow, wrist)
                        j2c, j3c = self._apply_calibration(j2r, j3r)
                        
                        if self._check_limits(j2c, j3c):
                            self._ema(j2c, j3c)
                            self._has_detection = True
                            self._limits_ok = True
                            color = arm_color
                        else:
                            self._has_detection = True
                            self._limits_ok = False
                            color = (0, 0, 255)

                        # Draw skeleton
                        s_pt = (int(shoulder[0]), int(shoulder[1]))
                        e_pt = (int(elbow[0]),    int(elbow[1]))
                        w_pt = (int(wrist[0]),    int(wrist[1]))
                        cv2.line(frame, s_pt, e_pt, color, 3)
                        cv2.line(frame, e_pt, w_pt, color, 3)
                        for pt in (s_pt, e_pt, w_pt):
                            cv2.circle(frame, pt, 5, color, -1)

                        self._last_good_t = now

            elif now - self._last_good_t < 0.5:
                pass  # Keep last state

            # Rate-limited command publishing
            if (now - self._last_cmd_t >= self._cmd_interval
                    and self._has_detection and self._limits_ok):
                j2d = math.degrees(self._j2_smooth)
                j3d = math.degrees(self._j3_smooth)

                if (self._last_j2_sent is None
                        or abs(j2d - self._last_j2_sent) > MIN_ANGLE_DEG
                        or abs(j3d - self._last_j3_sent) > MIN_ANGLE_DEG):
                    self._publish_cmd(self._j2_smooth, self._j3_smooth)
                    self._last_j2_sent, self._last_j3_sent = j2d, j3d
                    self._last_cmd_t = now

                    ts = time.strftime("%H:%M:%S")
                    self.get_logger().info(
                        f"{ts} [{self._cmds_sent:03d}] {self._arm_mode} | "
                        f"J2={j2d:+5.1f}° J3={j3d:+5.1f}° | Q={int(self._elbow_conf*100)}%"
                    )

            # Display
            self._draw_osd(frame)
            cv2.imshow(win, frame)

            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("👋 Quitting...")
                break
            elif key in (ord('h'), ord('H')):
                self._send_home()

            rclpy.spin_once(self, timeout_sec=0.001)

        self._cap.release()
        cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = SO101MimicNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
