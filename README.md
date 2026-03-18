# 🤖 lerobot_ws

A ROS 2 workspace for experimenting with the **SO-101** robotic arm using the
[LeRobot](https://github.com/huggingface/lerobot) ecosystem by Hugging Face.

This workspace builds on top of two key repositories:

- 🔧 **[lerobot-ros](https://github.com/ycheng517/lerobot-ros)** — a generic
  ROS 2 interface that connects any `ros2_control` / MoveIt-compatible robot arm
  with the LeRobot framework. Provides joint-position control, end-effector
  velocity control via MoveIt Servo, and teleoperator devices (keyboard & gamepad).

- 🏗️ **[lerobot_ws (Pavankv92)](https://github.com/Pavankv92/lerobot_ws)** —
  Gazebo simulation setup and `ros2_control` configuration for the SO-101 arm,
  used here as the simulation backbone.

---

## 📦 Workspace Structure
    src/
    ├── lerobot_ros/ # ROS 2 ↔ LeRobot bridge (based on ycheng517/lerobot-ros)
    ├── lerobot_sim/ # Gazebo simulation for SO-101 (based on Pavankv92/lerobot_ws)
    ├── lerobot_mimic/ # Mimic / imitation learning experiments
    └── lerobot_candy/ # Custom task experiments (e.g. candy picking)

## 🦾 About the Robot

The **SO-101** is a low-cost, open-source 6-DoF robotic arm designed for
imitation learning and manipulation research. It is the primary hardware target
of this workspace.

## 🚀 Projects Inside


| `lerobot_ros` | ROS 2 bridge: joint control, teleoperation, MoveIt integration |
| `lerobot_sim` | Gazebo simulation environment for the SO-101 arm |
| `lerobot_mimic` | Imitation learning experiments with the SO-101 |
| `lerobot_candy` | Custom task: robot picking candy using learned policies |

## 🛠️ Requirements

- ROS 2 Jazzy
- Python 3.12 (conda environment recommended)
- Gazebo (for simulation)
- MoveIt 2 (for end-effector control)
- LeRobot (Hugging Face)

## 📬 Feel free to explore, fork, and reach out!

