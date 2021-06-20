#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from bridge import test_sub




if __name__ == "__main__":

    rospy.init_node('bridge_sub_test')

    lid = test_sub()

    rospy.spin()