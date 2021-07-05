#!/usr/bin/env python3

import rospy
import torch 
import numpy as np
from sensor_msgs.msg import CompressedImage
from mr_cv.msg import OutputCV
from geometry_msgs.msg import Twist

from utils.bridge import ImgBridge, OutputCVBridge

from actions.follower import BoxFollower


class Actuator:

    def __init__(self, action):

        action_dict = {'follow':BoxFollower}

        cv_sub = rospy.Subscriber('/filtered_output', OutputCV, self.cv_cb)

        self.follower = action_dict[action]()

        self.bridge = OutputCVBridge()

        self.last_detected = None

        self.reuses = 0


    def cv_cb(self, msg):
        
        detection, img_resolution, label_list = self.bridge.outputcv_to_np(msg)
        print(img_resolution)
        box = detection['boxes'][0]
        print(box)
        self.follower.follow(box, img_resolution)

    
    


if __name__ == '__main__':

    rospy.init_node('actuator')

    # action = str(rospy.get_param('~action'))
    action = 'follow'
    actuator = Actuator(action)

    rospy.spin()


