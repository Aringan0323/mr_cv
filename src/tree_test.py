#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from behavior_tree.nodes import Selector, Sequencer, Action, Conditional


class Wander:

    def __init__(self):

        scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_cb)

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self.



