#!/usr/bin/env python3

import rospy
import numpy as np
import torch
import cv2


def box_centroid(box, img_resolution):

    cX = (box[0]+box[2])/2

    cY = (box[1]+box[3])/2

    return (cX, cY)




def mask_centroid(mask):

    np_mask = mask.cpu().numpy()[15]
    print(np_mask.shape)
    M = cv2.moments(np_mask)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return (cX, cY)

