#!/usr/bin/env python3

import torch
import rospkg

from abc import abstractmethod


class Model(object):

    def __init__(self):

        # All tensors and tensor operations will be performed on the GPU if one is available.
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # Use rospack to navigate filesystem
        self.rospack = rospkg.RosPack()
        
        # Define the transform and pytorch model
        self.trf = identity
        self.model = identity

        # Create a dictionary mapping between label IDs and category names.
        # Example: self.labels_dict = {0 : "background"}
        self.labels_dict = {}


    @staticmethod
    def identity(input):
        return input


    @abstractmethod
    def forward(self, img):
        
        inpt = self.trf(img)
        output = self.model(inpt)
        return output
        