#!/usr/bin/env python3

import sys
sys.path.append("..")

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage


import numpy as np

from nodes import Conditional, Action, Sequencer, Selector

from action_nodes.basic_movement import *

from update_nodes.basic_updates import *
from update_nodes.movement_control_updates import *
from update_nodes.cv_updates import *
from update_nodes.scan_updates import *

from conditional_nodes.scan_conditionals import *
from conditional_nodes.basic_conditionals import *

from ros_behavior_tree import ROSBehaviorTree


def WallAvoider(wall_dist=0.5, front_fov=50, turn_vel=0.3):

    wall_ahead = WallAhead(wall_dist, wall_fov)

    moving_forward = BoolVar('moving forward')
    
    def custom_stop():

        moving_forward = BoolVar('moving forward')
        stop = Stop()
        update = FlipBoolVar('moving forward')

        seq = Sequencer([moving_forward,stop,update])

        return seq

    stop = custom_stop()

    turn = AngularStatic(turn_vel)

    stop_and_turn = Selector([stop, turn])

    turn_away = Sequencer([wall_ahead, stop_and_turn])

    def custom_forward():

        not_moving_forward = BoolVarNot('moving forward')
        forward = LinearStatic(.2)
        moving_forward = FlipBoolVar('moving forward')

        seq = Sequencer([not_moving_forward, forward, moving_forward])

        return seq

    move_forward = custom_forward()

    wander = Selector([turn_away, move_forward])

    return wander


def WallFollower(front_dist_fov=10, wall_follow_dist=0.05, lin_vel=0.1):

    

    nearest_wall_angle = CalcNearestWallAngle('/scan','nearest_wall_angle')

    nearest_dist = CalcNearestDist('/scan', 'nearest_dist')

    avg_front_dist = CalcAvgFrontDist('/scan', 'avg_front_dist', 10)

    offset = 3.14159/2
    angular_pid = AngularPID('angular_pid', 'nearest_dist', 'nearest_wall_angle', 0.7, 0.7, 0.8, wall_follow_dist, offset=offset)

    linear_pid = LinearPID('linear_pid', 'avg_front_dist', lin_vel, offset=wall_follow_dist)

    move = LinearAngularDynamic('linear_pid', 'angular_pid')

    wall_follower = Sequencer([nearest_wall_angle, nearest_dist, avg_front_dist, angular_pid, linear_pid, move])

    return wall_follower


def ItemFollower(item="person", stopper="bottle", stopping_dist=0.07, stopping_fov=30, cam_res=[410,308]):

    clear_ahead = ClearAhead(stopping_dist, stopping_fov)

    detect = FastDetector('detector_label_dict', '/raspicam_node/image/compressed', 'detection')

    # item_err_var_name, label_dict_var_name, item_var_name, detection_var_name, camera_resolution
    stopper = ItemBearingErr('_', 'detector_label_dict', stopper, 'detection', cam_res)
    stop = Stop()
    saw_stopper = Sequencer([stopper, stop])

    item_bearing = ItemBearingErr('item_err', 'detector_label_dict', item, 'detection', cam_res)
    move = LinearAngularDynamic('static_lin','item_err')
    item_clear = Sequencer([item_bearing, move])

    check_follow = Selector([saw_stopper, item_clear])

    follow_item = Sequencer([clear_ahead, detect, check_follow])

    wall_follower = WallFollower()

    item_follower = Selector([follow_item, wall_follower])

    return item_follower


if __name__ == "__main__":

    rospy.init_node('person_follower')

    blackboard_vars = [
                    ['/scan', LaserScan],
                    ['/raspicam_node/image/compressed', CompressedImage],
                    ['detector_label_dict', None],
                    ['detection', None],
                    ['static_lin', 0.1],
                    ['item_err', None],
                    ['nearest_wall_angle', None],
                    ['nearest_dist', None],
                    ['avg_front_dist', None],
                    ['angular_pid', None],
                    ['linear_pid', None],
                    ['__', None]
                    ]

    tree = ROSBehaviorTree(ItemFollower(item='person', stopper='bottle'), blackboard_vars, print_vars=[])
    rospy.spin()

