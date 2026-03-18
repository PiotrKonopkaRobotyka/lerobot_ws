# 🦾 lerobot_mimic

A ROS 2 node (**ROS 2 Jazzy**) that makes the **SO-101 robotic arm** mirror
the operator's right arm in real time — using only a webcam and
**YOLOv26 pose estimation**.

---

## 🎥 How It Works

```
Webcam → YOLOv26 Pose → Keypoint Extraction
       → Angle Computation → EMA Smoothing
       → Safety Limit Check → JointTrajectory → SO-101
```


---

## 🦴 Joint Mapping

Only **J2** and **J3** are driven by pose estimation. Others are fixed:

    | Joint | Name | Control | Value |
    |---|---|---|---|
    | J1 | `shoulder_pan` | Fixed | `0.0 rad` |
    | J2 | `shoulder_lift` | **CV-driven** | From pose ±100° |
    | J3 | `elbow_flex` | **CV-driven** | From pose ±90° |
    | J4 | `wrist_flex` | Fixed | `0.0 rad` |
    | J5 | `wrist_roll` | Fixed | `0.0 rad` |

---

## 🖥️ OSD Overlay (Camera Window)

    | Element | Meaning |
    |---|---|
    | 🟢 `ACTIVE` | Tracking pose, angles within limits |
    | 🔴 `UNSAFE` | Pose detected but joint limit exceeded |
    | 🟡 `SEARCH` | No person detected |
    | `J2 / J3` | Live smoothed angles in degrees |
    | `Quality %` | Elbow keypoint confidence (green ≥ 70%) |
    | `FPS` | Live processing frame rate |
    | `Cmds` | Total trajectory commands sent |
    | `SIGN J2/J3` | Active calibration polarity reminder |

---

## 🎮 Keyboard Controls

    | Key | Action |
    |---|---|
    | `H` | Send robot to **home position** (all joints = 0°, 2s move) |
    | `Q` | Quit the node cleanly |

---

## 🛠️ Requirements

```bash
# Python dependencies
pip install ultralytics opencv-python numpy

# ROS 2 system package
sudo apt install ros-jazzy-trajectory-msgs
```

- **ROS 2 Jazzy**
- `ros2_control` with `joint_trajectory_controller` on `/arm_controller/joint_trajectory`
- Webcam at `/dev/video0`

---

## 🚀 Running

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash

python3 src/lerobot_mimic/scripts/lerobot_mimic.py
```

> ⚠️ Start the Gazebo simulation (or real robot) and ensure `arm_controller`
> is active **before** launching this node.

---

## 🔧 Calibration

If the robot moves in the **wrong direction**, edit the top of the script:

```python
J2_SIGN = +1   # flip if shoulder moves backwards
J3_SIGN = +1   # flip if elbow moves backwards
J2_OFFSET = -1.5708   # adjust zero-point [rad]
J3_OFFSET = -1.5708   # adjust zero-point [rad]
```

The node holds the last valid pose for **0.5 s** after detection is lost —
preventing sudden drops when the arm briefly leaves the camera frame.
