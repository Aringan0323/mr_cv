#!/usr/bin/env python3

import rospy
import torch 
import numpy as np
from sensor_msgs.msg import CompressedImage
from mr_cv.msg import OutputCV32
from mr_cv.msg import OutputCV64
from geometry_msgs.msg import Twist

from utils.bridge import ImgBridge, OutputCVBridge
from utils.out_processing import box_centroid, mask_centroid


class Actuator:

    def __init__(self, item):

        cv_sub = rospy.Subscriber('/output', OutputCV32, self.cv_cb)

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self.twist = Twist()

        self.bridge = OutputCVBridge()

        self.last_detected = None

        self.reuses = 0

        self.item = item


    def cv_cb(self, msg):
        
        raw_detection, img_resolution, label_list = self.bridge.outputcv_to_torch(msg)
        print(raw_detection.shape)
        # if isinstance(raw_detection, dict):
        #     item_boxes = raw_detection['boxes'][raw_detection['labels']==label_list.index(self.item)]
        #     item_scores = raw_detection['scores'][raw_detection['labels']==label_list.index(self.item)]
        #     detection = {'boxes':item_boxes, 'scores':item_scores}
        #     cX, cY = 


        cX, cY = mask_centroid(raw_detection[15], 0.8)
        print(cX, cY)
        err = self.pid_err(cX, img_resolution)

        self.twist.angular.z = 1*err
        if err == 0:
            self.twist.linear.x = 0
        else:
            self.twist.linear.x = 0.2


        self.cmd_vel_pub.publish(self.twist)

    
    def pid_err(self, cX, img_resolution):

        xres_half = img_resolution[0]/2

        err = cX-xres_half

        return err


if __name__ == '__main__':

    rospy.init_node('actuator')

    item = str(rospy.get_param('~item'))

    actuator = Actuator(item)

    rospy.spin()


