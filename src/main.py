#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class lidar_tester:

    def __init__(self):

        self.sub = rospy.Subscriber('/scan', LaserScan, self.scan_cb)

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self.twist = Twist()



    def scan_cb(self, msg):
        
        self.ranges = msg.ranges
        print(self.ranges)


    


if __name__ == "__main__":

    rospy.init_node('lidar_test')

    lid = lidar_tester()

    rospy.spin()