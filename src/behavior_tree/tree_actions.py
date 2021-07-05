#!/usr/bin/env python3

import rospy
import numpy as np
import time

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from nodes import Selector, Sequencer, Action, Conditional


def filter_scan(msg):

    ranges = np.array(msg.ranges)

    range_min = msg.range_min
    range_max = msg.range_max
    ranges[ranges < range_min] = 99
    ranges[ranges > range_max] = 99

    num_ranges = ranges.size

    return ranges, num_ranges

    


def detect_wall_cb(blackboard):

    ranges, num_ranges = filter_scan(blackboard['/scan'])

    front_right = ranges[0:int(num_ranges/16)]
    front_left = ranges[num_ranges-int(num_ranges/16)]

    if np.min(front_right) > 0.2 and np.min(front_left) > 0.2:
        return False
    else:
        return True

def detect_wall(front_range):

    return Action(detect_wall_cb)



class TurnAwayCB:

    def __init__(self):

        self.twist = Twist()

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

    def move(self, blackboard)


    def turn(self, blackboard):

        ranges, num_ranges = filter_scan(blackboard['/scan'])

        min_angle = np.argmin(ranges)

        back = num_ranges/2

        diff = min_angle-back

        if abs(diff) > 5:

            if diff < 0:
                self.twist.angular.z = 1
            else:
                self.twist.angular.z = -1
            
            self.pub.publish(self.twist)
            return 'running'
        else:
            self.twist.angular.z = 0
            
            return 'success'





def turn_away_from_wall():

    cb = TurnAwayCB()

    return Action(cb.turn)




def move_cb(blackboard):

