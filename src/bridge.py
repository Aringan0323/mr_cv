#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2


def imgmsg_to_np(img_msg):
    # Since CvBride is trash, here is a way to directly convert a compressed image
    # message to a numpy array.

    img = np.fromstring(img_msg.data, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    return img


def np_to_imgmsg(img):
    # Again, since CvBride is trash, here is a way to directly convert a numpy array
    # to a compressed image message.

    imgmsg = CompressedImage()
    imgmsg.header.stamp = rospy.Time.now()
    imgmsg.format = "jpg"
    _, img_enc = cv2.imencode('.jpg', img)
    imgmsg.data = np.array(img_enc).tostring()

    return imgmsg