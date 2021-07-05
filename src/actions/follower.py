#!/usr/bin/env python3

import rospy
import numpy as np
import torch
from .action_utils.centroid import box_centroid
from .action_utils.pid_err import pid_err

from geometry_msgs.msg import Twist


# All followers must work with numpy arrays. Use of the files in action_utils is encouraged
class BoxFollower:

    def __init__(self):

        self.twist = Twist()

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    
    def follow(self, box, img_resolution):
        if box.size > 0:
            cX, cY = box_centroid(box)
            x_err, _ = pid_err(cX, cY, img_resolution)
            print(cX, cY)
        else:
            x_err = 0
        self.twist.angular.z = 1*x_err
        if x_err == 0:
            self.twist.linear.x = 0
        else:
            self.twist.linear.x = 0.2


        self.pub.publish(self.twist)