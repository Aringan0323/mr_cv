#!/usr/bin/env python3

import rospy
import numpy as np

class ScanFilter:

    def __init__(self, left_bearing, right_bearing):

        