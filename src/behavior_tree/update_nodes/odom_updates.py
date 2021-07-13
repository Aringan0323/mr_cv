#!/usr/bin/env python3

import sys
sys.path.append("..") # Adds higher directory to python modules path.

import rospy
import numpy as np 
import math
from nav_msgs.msg import Odometry

from nodes import Update


class GetRotation(Update):


    def __init__(self, rotation_var_name, degrees=True):

        self.rotation_var_name = rotation_var_name


        if degrees:
            # Conversion between radians and degrees
            self.mult = 180/3.1415
        else:
            self.mult = 1


    def tick(self, blackboard):

        try:

            # Getting orientation from msg
            ori = blackboard['/odom'].msg.pose.pose.orientation
            x = ori.x
            y = ori.y
            z = ori.z
            w = ori.w

            # Converting the quaternion values to radians, and then to degrees if the user specifies
            t3 = 2.0 * (w * z + x * y)
            t4 = 1.0 - 2.0 * (y * y + z * z)
            rot = math.atan2(t3, t4)

            blackboard[self.rotation_var_name] = rot * self.mult

            return "success"

        except:

            return "failure"


