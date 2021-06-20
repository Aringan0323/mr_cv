#!/usr/bin/env python3


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch
import torchvision
from time import time



class Detector:

    def __init__(self):

        self.type = 