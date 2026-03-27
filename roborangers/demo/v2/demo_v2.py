#!/usr/bin/env python3

###############################################
#               C O M M A N D S               #
###############################################

'''
ros2 service call /rob498_drone_6/comm/land std_srvs/srv/Trigger {}
ros2 service call /rob498_drone_6/comm/launch std_srvs/srv/Trigger {}
ros2 service call /rob498_drone_6/comm/abort std_srvs/srv/Trigger {}
ros2 service call /rob498_drone_6/comm/test std_srvs/srv/Trigger {}

ros2 topic echo /rob498_drone_6/comm/waypoints

header:
  stamp:
    sec: 1772809884
    nanosec: 428233609
  frame_id: vicon/world
poses:
- position:
    x: -2.3
    y: 2.3
    z: 0.5
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0
- < more poses >

'''

import rclpy
from comm_node import CommNode

###############################################
#              M A I N   L O O P              #
###############################################

def main(args=None):
    rclpy.init(args=args)
    node = CommNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
