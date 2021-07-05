#!/usr/bin/env python3

import rospy
import numpy as np
import torch
import cv2


def box_centroid(box):

    cX = (box[0]+box[2])/2

    cY = (box[1]+box[3])/2

    return (cX, cY)



# Should calculate the centroid of a mask. Does not yet work
def mask_centroid(mask):

    M = cv2.moments(mask)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return (cX, cY)

