import sys
from lerobot.scripts.lerobot_teleoperate import main

# Importujemy nasze modyfikacje, aby dekoratory @RobotConfig.register_subclass z ycheng517 się odpaliły
import lerobot_robot_ros
import lerobot_teleoperator_devices

if __name__ == "__main__":
    # Nadpisujemy argumenty systemowe (to samo co wpisywałeś w bashu)
    sys.argv = [
        "lerobot-teleoperate",
        "--robot.type=so101_ros",
        "--robot.id=my_awesome_follower_arm",
        "--teleop.type=keyboard_joint",
        "--teleop.id=my_awesome_leader_arm",
        "--display_data=true"
    ]
    main()
