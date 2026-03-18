#!/usr/bin/env python3
"""
SO-101 Pose Mimic Node
======================
ROS 2 node (Jazzy) that reads a webcam, runs YOLO pose estimation,
and publishes JointTrajectory commands so the SO-101 arm mirrors
the operator's right arm in real time.

Dependencies:
    pip install ultralytics opencv-python numpy
    ros-jazzy-trajectory-msgs  (sudo apt install)

Usage:
    python3 so101_mimic.py
    Keys: H → HOME   Q → quit
"""

# ── Standard library ────────────────────────────────────────────────────────
import contextlib   # context manager used to silently suppress C-level stderr
import math         # atan2, sin, cos, degrees, radians
import os           # low-level file-descriptor duplication for stderr redirect
import sys          # sys.exit() on fatal startup errors
import time         # wall-clock timing for FPS, rate limiting, timestamps

# ── Third-party ─────────────────────────────────────────────────────────────
import cv2          # camera capture, image drawing, window management
import numpy as np  # array math, NaN detection on keypoint arrays

from ultralytics import YOLO  # YOLOv8/v11 pose estimation model

# ── ROS 2 ────────────────────────────────────────────────────────────────────
import rclpy                                          # ROS 2 Python client library
from rclpy.node import Node                           # base class for all ROS 2 nodes
from trajectory_msgs.msg import (                     # joint motion message types
    JointTrajectory,
    JointTrajectoryPoint,
)


# ===========================================================================
# ⚙️  CALIBRATION — adjust these values if the arm moves in the wrong direction
# ===========================================================================

J2_SIGN:   int   = -1        # shoulder-lift axis polarity: -1 or +1
J3_SIGN:   int   = -1        # elbow-flex axis polarity:    -1 or +1
J2_OFFSET: float = -1.5708   # shoulder-lift zero-point shift [rad] ≈ -π/2
J3_OFFSET: float = -1.5708   # elbow-flex zero-point shift    [rad] ≈ -π/2

# Safe resting configuration sent on startup and when 'H' is pressed
HOME_POSITION: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0]  # [J1..J5] in radians


# ===========================================================================
# ⚙️  ROBOT / CONTROL CONFIGURATION
# ===========================================================================

JOINT_NAMES: list[str] = ["1", "2", "3", "4", "5"]  # must match controller YAML
YOLO_MODEL:  str        = "yolo26m-pose.pt"           # pose model weights file
CONF_THRESHOLD: float   = 0.3    # minimum keypoint confidence accepted (0–1)
MIN_ANGLE_DEG:  float   = 2.0    # deadband: suppress command if change < 2 °
CONTROL_HZ:     int     = 10     # maximum trajectory publish rate [Hz]
TRAJ_TIME_S:    float   = 0.08   # time budget allocated per trajectory point [s]
EMA_ALPHA:      float   = 0.5    # EMA smoothing factor (0 = slow, 1 = no filter)

# SO-101 hardware joint limits [rad] — taken from URDF / datasheet
J1_LIM = (-1.919,  1.919)   # shoulder pan   ≈ ±110°
J2_LIM = (-1.74,   1.74)    # shoulder lift  ≈ ±100°
J3_LIM = (-1.5708, 1.5708)  # elbow flex     ≈  ±90°
J4_LIM = (-1.65,   1.65)    # wrist flex     ≈  ±94°
J5_LIM = (-2.74,   2.84)    # wrist roll     ≈ ±157° — not used in mimic mode

# Degree equivalents precomputed once for limit checks and OSD rendering
J2_LIM_DEG = (math.degrees(J2_LIM[0]), math.degrees(J2_LIM[1]))
J3_LIM_DEG = (math.degrees(J3_LIM[0]), math.degrees(J3_LIM[1]))

# Joints not driven by pose estimation — held at fixed values
FIXED_J1: float = 0.0   # shoulder pan  — kept forward-facing
FIXED_J4: float = 0.0   # wrist flex    — kept neutral
FIXED_J5: float = 0.0   # wrist roll    — kept neutral

# YOLO COCO keypoint indices
# NOTE: image is horizontally flipped, so left body side appears on the right
KP_L_SHOULDER = 5   # left shoulder keypoint index (appears as "right" on screen)
KP_L_ELBOW    = 7   # left elbow    keypoint index
KP_L_WRIST    = 9   # left wrist    keypoint index

# Camera capture resolution
CAM_W: int = 854   # desired capture width  [px]
CAM_H: int = 480   # desired capture height [px]


# ===========================================================================
# Utility helpers
# ===========================================================================

@contextlib.contextmanager
def _suppress_stderr():
    """Redirect file-descriptor 2 to /dev/null for the duration of the block.

    Required because YOLO prints a C-level banner that bypasses Python logging.
    """
    with open(os.devnull, "w") as devnull:   # open the null device for writing
        saved_fd = os.dup(2)                  # save original stderr file descriptor
        os.dup2(devnull.fileno(), 2)          # point fd-2 at /dev/null
        try:
            yield                             # run the wrapped block silently
        finally:
            os.dup2(saved_fd, 2)             # restore original stderr
            os.close(saved_fd)               # release the duplicate descriptor


def _normalize_angle(angle: float) -> float:
    """Wrap *angle* into the open interval (−π, π] using the atan2 identity."""
    return math.atan2(math.sin(angle), math.cos(angle))  # single-call normalisation


# ===========================================================================
# Main ROS 2 Node
# ===========================================================================

class SO101MimicNode(Node):
    """ROS 2 node: webcam → YOLO keypoints → calibrated angles → JointTrajectory."""

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__("so101_mimic")                    # register node name in the ROS 2 graph

        # Arm-controller publisher — QoS depth 10 is sufficient for 10 Hz
        self._pub = self.create_publisher(
            JointTrajectory,
            "/arm_controller/joint_trajectory",
            10,                                            # message queue depth
        )

        self._load_model()    # step 1: load YOLO weights
        self._open_camera()   # step 2: open video device and verify first frame

        # ── Pose / control state ─────────────────────────────────────────
        self._arm_mode:      str   = "RIGHT"  # arm side to track (only RIGHT supported)
        self._j2_smooth:     float = 0.0      # EMA-filtered shoulder-lift angle [rad]
        self._j3_smooth:     float = 0.0      # EMA-filtered elbow-flex    angle [rad]
        self._has_detection: bool  = False    # True when a valid skeleton was found
        self._limits_ok:     bool  = True     # True when last angles are within limits
        self._last_good_t:   float = 0.0      # wall time of the last valid detection
        self._elbow_conf:    float = 0.0      # YOLO confidence of the elbow keypoint

        # ── Rate-limiter state ────────────────────────────────────────────
        self._cmd_interval: float        = 1.0 / CONTROL_HZ  # minimum seconds between publishes
        self._last_cmd_t:   float        = 0.0               # wall time of the last publish
        self._last_j2_sent: float | None = None              # last sent J2 value [°] for deadband
        self._last_j3_sent: float | None = None              # last sent J3 value [°] for deadband
        self._cmds_sent:    int          = 0                 # total published commands (OSD counter)

        # ── Performance counters ──────────────────────────────────────────
        self._frame_count: int   = 0             # total frames processed since start
        self._start_t:     float = time.time()   # node start time used for FPS calculation

        self._print_help()   # log configuration banner and key bindings to the console

    # -----------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the YOLO pose model, suppressing noisy C-level startup output."""
        self.get_logger().info(f"⏳ Loading {YOLO_MODEL}...")
        with _suppress_stderr():            # hide YOLO's internal C++ banner
            self._model = YOLO(YOLO_MODEL)  # parse and load model weights from file
        self.get_logger().info("✓ YOLO loaded")

    # -----------------------------------------------------------------------

    def _open_camera(self) -> None:
        """Open /dev/video0, set resolution, and verify that frames are readable."""
        self._cap = cv2.VideoCapture(0)          # open the default camera index
        if not self._cap.isOpened():             # abort immediately if the device is unavailable
            self.get_logger().fatal("❌ Cannot open camera!")
            sys.exit(1)

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)  # request desired frame width
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)  # request desired frame height
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)         # single-frame buffer minimises latency

        ret, frame = self._cap.read()            # attempt to read one test frame
        if not ret:                              # abort if the camera is not streaming
            self.get_logger().fatal("❌ Cannot read camera!")
            sys.exit(1)
        self.get_logger().info(f"✓ Camera {frame.shape[1]}x{frame.shape[0]}")

    # -----------------------------------------------------------------------

    def _print_help(self) -> None:
        """Log a startup banner that shows configuration values and keyboard shortcuts."""
        lines = [
            "═" * 55,
            "SO-101 Pose Mimic — CALIBRATED",
            f"  Model   : {YOLO_MODEL}",
            f"  Conf    : {CONF_THRESHOLD}",
            f"  Deadband: {MIN_ANGLE_DEG}°",
            f"  TrajTime: {TRAJ_TIME_S}s",
            f"  J2_SIGN : {J2_SIGN}   J3_SIGN: {J3_SIGN}",
            "─" * 55,
            "  H     — send robot to HOME position",
            "  Q     — quit node",
        ]
        for line in lines:
            self.get_logger().info(line)   # each line is a separate log entry

    # -----------------------------------------------------------------------
    # Angle computation
    # -----------------------------------------------------------------------

    @staticmethod
    def _compute_angles_raw(
        shoulder: np.ndarray,
        elbow:    np.ndarray,
        wrist:    np.ndarray,
    ) -> tuple[float, float]:
        """Compute raw J2 (shoulder lift) and J3 (elbow flex) from 2-D pixel coordinates.

        Coordinate convention:
          • Image X increases rightward.
          • Image Y increases downward — vectors that point "up" have negative image-Y.
          • Math Y is flipped (ay = shoulder.y − elbow.y) so that atan2 gives intuitive angles.

        Returns:
            (j2_raw, j3_raw) both normalised to (−π, π].
        """
        # ── J2: angle of the upper-arm segment from the +X axis ──────────
        dx = elbow[0] - shoulder[0]    # upper-arm horizontal displacement [px]
        dy = elbow[1] - shoulder[1]    # upper-arm vertical displacement   [px] (image Y-down)
        j2 = math.atan2(dy, dx)        # angle of upper-arm vector in image space
        j2 = _normalize_angle(j2)      # wrap to (−π, π]

        # ── J3: relative angle between upper-arm and forearm ─────────────
        ax = elbow[0]  - shoulder[0]   # upper-arm vector X (image coords)
        ay = shoulder[1] - elbow[1]    # upper-arm vector Y (flipped to math Y-up)
        bx = wrist[0]  - elbow[0]      # forearm vector X
        by = elbow[1]  - wrist[1]      # forearm vector Y (flipped to math Y-up)

        ang1 = math.atan2(ay, ax)      # absolute angle of the upper-arm segment
        ang2 = math.atan2(by, bx)      # absolute angle of the forearm segment
        j3   = ang1 - ang2             # relative bend angle at the elbow
        j3   = _normalize_angle(j3)    # wrap to (−π, π]

        return j2, j3

    # -----------------------------------------------------------------------

    @staticmethod
    def _apply_calibration(j2_raw: float, j3_raw: float) -> tuple[float, float]:
        """Flip axis direction and shift the zero-point, then re-normalise to (−π, π].

        The sign constants and offsets at the top of the file map the image-space
        angles onto the SO-101 URDF joint conventions.
        """
        j2 = J2_SIGN * j2_raw + J2_OFFSET   # apply polarity flip and zero-shift for J2
        j3 = J3_SIGN * j3_raw + J3_OFFSET   # apply polarity flip and zero-shift for J3
        j2 = _normalize_angle(j2)            # re-normalise after offset may push beyond ±π
        j3 = _normalize_angle(j3)            # re-normalise after offset may push beyond ±π
        return j2, j3

    # -----------------------------------------------------------------------
    # Safety
    # -----------------------------------------------------------------------

    def _check_limits(self, j2: float, j3: float) -> bool:
        """Return True if both angles are within hardware limits; log a warning otherwise."""
        j2d = math.degrees(j2)                              # convert to degrees for human-readable log
        j3d = math.degrees(j3)

        j2_ok = J2_LIM_DEG[0] <= j2d <= J2_LIM_DEG[1]    # check shoulder-lift range
        j3_ok = J3_LIM_DEG[0] <= j3d <= J3_LIM_DEG[1]    # check elbow-flex range

        if not (j2_ok and j3_ok):                          # at least one joint is out of range
            self.get_logger().warn(
                f"⚠️ LIMIT  J2={j2d:+.1f}° {J2_LIM_DEG}  J3={j3d:+.1f}° {J3_LIM_DEG}"
            )
        return j2_ok and j3_ok

    # -----------------------------------------------------------------------
    # Smoothing
    # -----------------------------------------------------------------------

    def _ema(self, j2_new: float, j3_new: float) -> None:
        """Apply one EMA step to the smoothed J2 and J3 state variables.

        Formula: smooth ← α·new + (1−α)·smooth
        """
        self._j2_smooth = EMA_ALPHA * j2_new + (1.0 - EMA_ALPHA) * self._j2_smooth  # EMA update J2
        self._j3_smooth = EMA_ALPHA * j3_new + (1.0 - EMA_ALPHA) * self._j3_smooth  # EMA update J3

    # -----------------------------------------------------------------------
    # Publishing
    # -----------------------------------------------------------------------

    def _publish_cmd(self, j2: float, j3: float) -> None:
        """Assemble and publish a single-waypoint JointTrajectory to the arm controller."""
        msg = JointTrajectory()                                    # empty trajectory envelope
        msg.joint_names = JOINT_NAMES                              # joint name list must match YAML

        pt = JointTrajectoryPoint()                                # single waypoint object
        pt.positions = [FIXED_J1, float(j2), float(j3),           # full 5-joint position vector
                        FIXED_J4, FIXED_J5]
        pt.time_from_start.sec     = 0                            # integer-seconds part of duration
        pt.time_from_start.nanosec = int(TRAJ_TIME_S * 1e9)       # fractional-nanoseconds part

        msg.points = [pt]              # trajectory consists of exactly one waypoint
        self._pub.publish(msg)         # send to /arm_controller/joint_trajectory
        self._cmds_sent += 1           # increment OSD command counter

    # -----------------------------------------------------------------------

    def _send_home(self) -> None:
        """Publish the HOME position with a slow 2-second execution time for safety."""
        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES

        pt = JointTrajectoryPoint()
        pt.positions               = list(HOME_POSITION)   # copy to prevent accidental mutation
        pt.time_from_start.sec     = 2                     # 2-second execution — deliberately slow
        pt.time_from_start.nanosec = 0

        msg.points = [pt]
        self._pub.publish(msg)                                           # publish home command

        deg = [f"{math.degrees(a):.1f}°" for a in HOME_POSITION]       # human-readable log
        self.get_logger().info(
            f"🏠 HOME → J1={deg[0]} J2={deg[1]} J3={deg[2]} J4={deg[3]} J5={deg[4]}"
        )

    # -----------------------------------------------------------------------
    # Keypoint extraction
    # -----------------------------------------------------------------------

    def _extract_keypoints(
        self, results
    ) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray | None, float]:
        """Parse YOLO results and return the three arm keypoints for the configured arm.

        Returns:
            (valid, shoulder, elbow, wrist, elbow_conf)
            *valid* is False when the person is absent, confidence is too low, or NaN appears.
        """
        if results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
            return False, None, None, None, 0.0     # no person detected in this frame

        kpts = results[0].keypoints.xy[0].cpu().numpy()    # (17, 2) array — all keypoints [px]
        conf = (
            results[0].keypoints.conf[0].cpu().numpy()     # (17,) confidence scores per keypoint
            if results[0].keypoints.conf is not None
            else None
        )

        if len(kpts) <= 10:                                 # need indices 0–10 at minimum
            return False, None, None, None, 0.0

        # Select keypoint indices for the configured arm (currently RIGHT / mirrored LEFT)
        si, ei, wi = KP_L_SHOULDER, KP_L_ELBOW, KP_L_WRIST   # shoulder, elbow, wrist indices

        shoulder   = kpts[si]                               # shoulder 2-D position [px]
        elbow      = kpts[ei]                               # elbow    2-D position [px]
        wrist      = kpts[wi]                               # wrist    2-D position [px]
        elbow_conf = float(conf[ei]) if conf is not None else 1.0  # elbow detection confidence

        valid = (
            elbow_conf >= CONF_THRESHOLD                            # elbow sufficiently visible
            and all(p[0] > 1 and p[1] > 1                          # no keypoint at pixel origin
                    for p in (shoulder, elbow, wrist))
            and not any(np.isnan(p).any()                          # no NaN values present
                        for p in (shoulder, elbow, wrist))
        )
        return valid, shoulder, elbow, wrist, elbow_conf

    # -----------------------------------------------------------------------
    # OSD overlay
    # -----------------------------------------------------------------------

    def _draw_osd(self, frame: np.ndarray) -> None:
        """Render status badge, joint angles, quality, FPS, and calibration onto *frame* in-place."""
        h, w = frame.shape[:2]
        if w < 200 or h < 100:   # skip OSD if the frame is unexpectedly tiny
            return

        # ── Status badge (top-left) ───────────────────────────────────────
        if self._has_detection and self._limits_ok:
            badge_color, label = (0, 200, 0), "ACTIVE"    # green  — tracking and safe
        elif self._has_detection:
            badge_color, label = (0, 0, 255), "UNSAFE"    # red    — out of limits
        else:
            badge_color, label = (0, 200, 200), "SEARCH"  # yellow — no person detected

        cv2.circle(frame, (20, 20), 10, badge_color, -1)                           # filled status dot
        cv2.putText(frame, label, (35, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)             # status label

        # ── Joint angle readout ───────────────────────────────────────────
        y = 55   # vertical text cursor — advanced after each rendered line
        if self._has_detection:
            j2d = math.degrees(self._j2_smooth)                    # smoothed J2 in degrees
            j3d = math.degrees(self._j3_smooth)                    # smoothed J3 in degrees
            angle_color = (0, 200, 0) if self._limits_ok else (0, 0, 255)   # green / red
            cv2.putText(frame, f"J2={j2d:+5.1f}  J3={j3d:+5.1f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, angle_color, 2)
            y += 25

            quality_pct   = int(self._elbow_conf * 100)                      # 0–100 %
            quality_color = (0, 200, 0) if quality_pct >= 70 else (0, 140, 255)  # green / orange
            cv2.putText(frame, f"Quality {quality_pct}%",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 2)
            y += 22

        # ── J3 limit reminder ─────────────────────────────────────────────
        cv2.putText(frame,
                    f"J3 lim [{J3_LIM_DEG[0]:.0f},{J3_LIM_DEG[1]:.0f}]",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

        # ── Bottom information bar ────────────────────────────────────────
        mode_color = (0, 220, 0) if self._arm_mode == "RIGHT" else (0, 220, 220)
        cv2.putText(frame, f"MODE {self._arm_mode}", (10, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 2)                 # arm mode

        elapsed = time.time() - self._start_t                  # seconds since node start
        fps = self._frame_count / elapsed if elapsed > 0 else 0.0
        cv2.putText(frame, f"FPS {fps:.1f}", (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)               # live FPS

        cv2.putText(frame, f"Cmds {self._cmds_sent}", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 2)             # total commands

        cv2.putText(frame, f"SIGN J2={J2_SIGN:+d} J3={J3_SIGN:+d}", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)           # calibration reminder

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self) -> None:
        """Blocking main loop: capture → infer → compute → publish → display → handle keys."""
        win = "SO-101 Mimic"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)   # create display window with auto-fit size

        time.sleep(0.5)       # allow ROS 2 graph to stabilise before first command
        self._send_home()     # move robot to a known-safe starting position

        self.get_logger().info("🚀 Main loop started")

        while rclpy.ok():                              # iterate until ROS 2 shuts down
            ret, frame = self._cap.read()              # fetch the latest camera frame
            if not ret:                                # abort if the camera stops providing frames
                self.get_logger().error("❌ Camera read failed!")
                break

            frame = cv2.flip(frame, 1)    # horizontal flip → mirror mode (natural to operator)
            self._frame_count += 1        # increment frame counter for FPS metric
            now = time.time()             # snapshot wall time for rate limiting

            # ── YOLO inference ────────────────────────────────────────────
            results = self._model(frame, verbose=False, conf=0.10)  # run pose estimator
            self._has_detection = False                              # reset detection flag each frame

            valid, shoulder, elbow, wrist, elbow_conf = self._extract_keypoints(results)
            self._elbow_conf = elbow_conf                           # store for OSD quality display

            if valid:
                # ── Angle pipeline ────────────────────────────────────────
                j2_raw, j3_raw = self._compute_angles_raw(shoulder, elbow, wrist)  # image-space
                j2_cal, j3_cal = self._apply_calibration(j2_raw, j3_raw)           # robot-space

                if self._check_limits(j2_cal, j3_cal):    # angles within hardware limits
                    self._ema(j2_cal, j3_cal)              # update smoothed state variables
                    self._has_detection = True             # valid and safe pose
                    self._limits_ok     = True
                    skel_color = (0, 220, 0)               # green skeleton — OK
                else:
                    self._has_detection = True             # detected but out of safe range
                    self._limits_ok     = False
                    skel_color = (0, 0, 255)               # red skeleton — UNSAFE

                # ── Draw arm skeleton ─────────────────────────────────────
                s_pt = (int(shoulder[0]), int(shoulder[1]))   # shoulder pixel position
                e_pt = (int(elbow[0]),    int(elbow[1]))       # elbow    pixel position
                w_pt = (int(wrist[0]),    int(wrist[1]))       # wrist    pixel position
                cv2.line(frame, s_pt, e_pt, skel_color, 3)    # upper-arm segment line
                cv2.line(frame, e_pt, w_pt, skel_color, 3)    # forearm  segment line
                for pt in (s_pt, e_pt, w_pt):
                    cv2.circle(frame, pt, 5, skel_color, -1)  # filled keypoint dot

                self._last_good_t = now    # record time of the last successful detection

            elif now - self._last_good_t < 0.5:
                pass    # hold the last known state for up to 0.5 s after detection is lost

            # ── Rate-limited trajectory publish ───────────────────────────
            interval_elapsed = (now - self._last_cmd_t) >= self._cmd_interval   # time gate
            pose_safe        = self._has_detection and self._limits_ok           # state gate

            if interval_elapsed and pose_safe:
                j2d = math.degrees(self._j2_smooth)    # smoothed J2 in degrees for deadband check
                j3d = math.degrees(self._j3_smooth)    # smoothed J3 in degrees for deadband check

                first_cmd = self._last_j2_sent is None                            # no command sent yet
                j2_moved  = first_cmd or abs(j2d - self._last_j2_sent) > MIN_ANGLE_DEG  # J2 changed enough
                j3_moved  = first_cmd or abs(j3d - self._last_j3_sent) > MIN_ANGLE_DEG  # J3 changed enough

                if j2_moved or j3_moved:                                          # at least one joint moved
                    self._publish_cmd(self._j2_smooth, self._j3_smooth)           # send trajectory
                    self._last_j2_sent = j2d                                      # update deadband baseline
                    self._last_j3_sent = j3d
                    self._last_cmd_t   = now                                      # reset rate-limiter

                    ts = time.strftime("%H:%M:%S")
                    self.get_logger().info(
                        f"{ts} [{self._cmds_sent:03d}] {self._arm_mode} | "
                        f"J2={j2d:+5.1f}° J3={j3d:+5.1f}° | Q={int(self._elbow_conf * 100)}%"
                    )

            # ── Render and display ────────────────────────────────────────
            self._draw_osd(frame)           # overlay all status information
            cv2.imshow(win, frame)          # show the annotated frame in the window

            # ── Keyboard handling ─────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF     # 1 ms poll keeps the loop non-blocking
            if key == ord('q'):
                self.get_logger().info("👋 Quitting...")
                break                       # exit the main loop cleanly
            elif key in (ord('h'), ord('H')):
                self._send_home()           # move robot to HOME when 'H' is pressed

            rclpy.spin_once(self, timeout_sec=0.001)   # process any pending ROS 2 callbacks

        # ── Cleanup ──────────────────────────────────────────────────────
        self._cap.release()          # release the camera device back to the OS
        cv2.destroyAllWindows()      # close all OpenCV display windows


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    """Initialise ROS 2, instantiate and run the mimic node, then shut down cleanly."""
    rclpy.init()                    # start the ROS 2 runtime
    node = SO101MimicNode()         # create node — loads YOLO and opens camera
    try:
        node.run()                  # enter the blocking main loop
    except KeyboardInterrupt:
        pass                        # Ctrl+C is a normal exit — no traceback needed
    finally:
        node.destroy_node()         # release all ROS 2 node resources
        rclpy.shutdown()            # shut down the ROS 2 runtime cleanly


if __name__ == "__main__":
    main()   # execute only when run directly, not when imported as a module
