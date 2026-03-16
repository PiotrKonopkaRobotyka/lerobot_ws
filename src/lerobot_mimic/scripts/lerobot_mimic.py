#!/usr/bin/env python3
"""
SO-101 Pose Mimic Node - CALIBRATED VERSION
Fixes:
  ✓ J2/J3 sign correction (image Y-down vs robot Y-up convention)
  ✓ Proper SO-101 joint limits (not MyCobot limits)
  ✓ Home position (H key) + manual offset tuning (+/- keys)
  ✓ EMA smoothing, confidence filter, proper trajectory timing
"""

import contextlib
import os
import cv2
import math
import numpy as np
import time
import sys

from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# ===========================================================================
# ⚙️  CALIBRATION CONSTANTS — zmień te wartości jeśli ruch jest odwrócony
# ===========================================================================
J2_SIGN   = -1      # Zmień na +1 jeśli shoulder idzie w złym kierunku
J3_SIGN   = -1      # Zmień na +1 jeśli elbow idzie w złym kierunku
J2_OFFSET = -1.5708      # Dodatkowy offset J2 [rad] - fine-tuning (+/- klawisze)
J3_OFFSET = -1.5708      # Dodatkowy offset J3 [rad]

# Home position — bezpieczna poza startowa [rad]
# J2=-0.5 (bark lekko do przodu), J3=1.0 (łokieć zgięty ~57°)
HOME_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0]  # [J1, J2, J3, J4, J5]

# ===========================================================================
# ⚙️  ROBOT CONFIG
# ===========================================================================
JOINT_NAMES    = ["1", "2", "3", "4", "5"]
YOLO_MODEL     = "yolo26m-pose.pt"
CONF_THRESHOLD = 0.5
MIN_ANGLE_DEG  = 3.0          # Deadband
CONTROL_HZ     = 10
TRAJ_TIME_S    = 0.5          # Trajectory execution time
EMA_ALPHA      = 0.25         # Smoothing (niższy = płynniej, ale wolniej)

# SO-101 JOINT LIMITS [rad] — z URDF / ECE4560
J1_LIM = (-1.919, 1.919)  # shoulder_pan
J2_LIM = (-1.74,  1.74)   # shoulder_lift
J3_LIM = (-1.69,  1.69)   # elbow_flex
J4_LIM = (-1.65,  1.65)   # wrist_flex
J5_LIM = (-2.74,  2.84)   # wrist_roll

# Fixed joints (nie sterowane przez CV)
FIXED_J1 = 0.0
FIXED_J4 = 0.0
FIXED_J5 = 0.0

# YOLO COCO keypoint indices
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW,    KP_R_ELBOW    = 7, 8
KP_L_WRIST,    KP_R_WRIST    = 9, 10

CAM_W, CAM_H = 854, 480
OFFSET_STEP  = 0.05   # krok regulacji offsetu [rad] ≈ 2.9°


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
        super().__init__("so101_mimic")

        self._pub = self.create_publisher(
            JointTrajectory, "/arm_controller/joint_trajectory", 10)

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
        self._j2_raw         = 0.0
        self._j3_raw         = 0.0
        self._j2_smooth      = 0.0
        self._j3_smooth      = 0.0
        self._has_detection  = False
        self._limits_ok      = True
        self._last_good_t    = 0.0
        self._elbow_conf     = 0.0
        self._current_j3_lim = (math.degrees(J3_LIM[0]), math.degrees(J3_LIM[1]))

        # Mutable calibration offsets (klawiatura)
        self._j2_offset = J2_OFFSET
        self._j3_offset = J3_OFFSET

        # Rate limiting
        self._cmd_interval  = 1.0 / CONTROL_HZ
        self._last_cmd_t    = 0.0
        self._last_j2_sent  = None
        self._last_j3_sent  = None
        self._cmds_sent     = 0

        # Perf
        self._frame_count = 0
        self._start_t     = time.time()

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
            "  R/L   — prawe/lewe ramię",
            "  H     — wyślij do HOME position",
            "  Q     — wyjście",
            "  ↑/↓   — J2 offset (+/-)",
            "  ←/→   — J3 offset (+/-)",
            "═" * 55,
        ]
        for l in lines:
            self.get_logger().info(l)

    # =======================================================================
    # Safety limits — SO-101 specific
    # =======================================================================
    def _check_limits(self, j2: float, j3: float) -> bool:
        j2d = math.degrees(j2)
        j3d = math.degrees(j3)
        j2_lo, j2_hi = math.degrees(J2_LIM[0]), math.degrees(J2_LIM[1])
        j3_lo, j3_hi = math.degrees(J3_LIM[0]), math.degrees(J3_LIM[1])

        j2_ok = j2_lo <= j2d <= j2_hi
        j3_ok = j3_lo <= j3d <= j3_hi
        self._current_j3_lim = (j3_lo, j3_hi)

        if not (j2_ok and j3_ok):
            self.get_logger().warn(
                f"⚠️ LIMIT  J2={j2d:+.1f}° [{j2_lo:.0f},{j2_hi:.0f}]  "
                f"J3={j3d:+.1f}° [{j3_lo:.0f},{j3_hi:.0f}]"
            )
        return j2_ok and j3_ok

    # =======================================================================
    # EMA filter
    # =======================================================================
    def _ema(self, j2_new: float, j3_new: float):
        self._j2_smooth = EMA_ALPHA * j2_new + (1 - EMA_ALPHA) * self._j2_smooth
        self._j3_smooth = EMA_ALPHA * j3_new + (1 - EMA_ALPHA) * self._j3_smooth

    # =======================================================================
    # Angle computation — poprawiona konwencja dla SO-101
    # =======================================================================
    @staticmethod
    def _compute_angles_raw(shoulder, elbow, wrist):
        """
        Obraz: Y rośnie w DÓŁ.
        Fizyczny świat: Y rośnie w GÓRĘ.
        Dlatego negujemy vy przy obliczaniu J2.
        """
        vx =  elbow[0] - shoulder[0]
        vy = -(elbow[1] - shoulder[1])  # ← negacja: flip Y (image → world)

        j2 = math.atan2(vy, vx)          # kąt ramienia od poziomej
        j2 = math.atan2(math.sin(j2), math.cos(j2))

        # Elbow angle: kąt między wektorem bark→łokieć a łokieć→nadgarstek
        ax, ay =  (elbow[0] - shoulder[0]),  (shoulder[1] - elbow[1])
        bx, by =  (wrist[0]  - elbow[0]),    (elbow[1]    - wrist[1])
        ang1 = math.atan2(ay, ax)
        ang2 = math.atan2(by, bx)
        j3 = ang1 - ang2
        j3 = math.atan2(math.sin(j3), math.cos(j3))

        return j2, j3

    # =======================================================================
    # Apply sign correction + offset
    # =======================================================================
    def _apply_calibration(self, j2_raw: float, j3_raw: float):
        j2 = J2_SIGN * j2_raw + self._j2_offset
        j3 = J3_SIGN * j3_raw + self._j3_offset
        # Normalize to (-π, π)
        j2 = math.atan2(math.sin(j2), math.cos(j2))
        j3 = math.atan2(math.sin(j3), math.cos(j3))
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

            q = int(self._elbow_conf * 100)
            qc = (0, 200, 0) if q >= 70 else (0, 140, 255)
            cv2.putText(frame, f"Quality {q}%",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, qc, 2); y += 22

        j3lo, j3hi = self._current_j3_lim
        cv2.putText(frame, f"J3 lim [{j3lo:.0f},{j3hi:.0f}]",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1); y += 20

        # Offsets (live tuning indicator)
        cv2.putText(frame,
            f"OFF J2={math.degrees(self._j2_offset):+.1f} J3={math.degrees(self._j3_offset):+.1f}",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)

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
                    else:
                        si, ei, wi = KP_R_SHOULDER, KP_R_ELBOW, KP_R_WRIST
                        arm_color  = (0, 220, 220)

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
                            self._j2_raw, self._j3_raw = j2r, j3r
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
            elif key in (ord('r'), ord('R')):
                self._arm_mode = "RIGHT"
                self.get_logger().info("➜ RIGHT ARM mode")
            elif key in (ord('l'), ord('L')):
                self._arm_mode = "LEFT"
                self.get_logger().info("➜ LEFT ARM mode")
            elif key in (ord('h'), ord('H')):
                self._send_home()
            elif key == 82:  # UP arrow
                self._j2_offset += OFFSET_STEP
                self.get_logger().info(f"⬆ J2_offset = {math.degrees(self._j2_offset):+.1f}°")
            elif key == 84:  # DOWN arrow
                self._j2_offset -= OFFSET_STEP
                self.get_logger().info(f"⬇ J2_offset = {math.degrees(self._j2_offset):+.1f}°")
            elif key == 83:  # RIGHT arrow
                self._j3_offset += OFFSET_STEP
                self.get_logger().info(f"➡ J3_offset = {math.degrees(self._j3_offset):+.1f}°")
            elif key == 81:  # LEFT arrow
                self._j3_offset -= OFFSET_STEP
                self.get_logger().info(f"⬅ J3_offset = {math.degrees(self._j3_offset):+.1f}°")

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
