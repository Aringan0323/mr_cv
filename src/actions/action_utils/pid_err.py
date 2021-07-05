#!/usr/bin/env python3

import rospy
import numpy as np
import torch
import cv2


def pid_err(cX, cY, img_resolution):

    xres_half = int(img_resolution[1]/2)
    yres_half = int(img_resolution[0]/2)

    x_err = xres_half-cX
    y_err = yres_half-cY

    return (-x_err, -y_err)