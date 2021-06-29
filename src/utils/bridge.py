#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from mr_cv.msg import OutputCV32
from mr_cv.msg import OutputCV64
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float64
from std_msgs.msg import MultiArrayDimension
import numpy as np
import torch
import cv2
import time

class ImgBridge:


    def __init__(self, format='jpg'):
        # Since CvBridge is shitty, this class is able to directly convert between 
        # numpy.ndarray objects and ROS CompressedImage messages.

        self.format = format

    
    def imgmsg_to_np(self, img_msg):
        
        img = np.fromstring(img_msg.data, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        return img


    def np_to_imgmsg(self, img):

        imgmsg = CompressedImage()
        imgmsg.header.stamp = rospy.Time.now()
        imgmsg.format = self.format
        _, img_enc = cv2.imencode('.'+self.format, img)
        imgmsg.data = np.array(img_enc).tostring()

        return imgmsg


class OutputCVBridge:


    def __init__(self, bit=32):
        # The bit parameter is by default 32 bit which means that the class will
        # convert torch tensors to OutputCV32 messages in the torch_to_outputcv()
        # function. Can be changed to 64 which will do the same but for OutputCV64
        # message types instead.

        self.bit = bit
        

    def torch_to_outputcv(self, output, img_resolution, label_list):
        # Takes in as input the detection output from either a box-detection or a 
        # segmentation pytorch model and converts it into an OutputCV ROS message.

        if self.bit == 32:
            outmsg = OutputCV32()
        elif self.bit == 64:
            outmsg = OutputCV64()
        else:
            print('{} bit message types not supported'.format(bit))
            return
        
        outmsg.img_resolution = img_resolution
        outmsg.label_list = label_list

        if isinstance(output, dict):
            outmsg.type = 'detection'

            if output['boxes'].is_cuda:
                boxes = output['boxes'].view(-1).cpu().tolist()
                scores = output['scores'].view(-1).cpu().tolist()
                labels = output['labels'].view(-1).cpu().tolist()
            else:
                boxes = output['boxes'].view(-1).tolist()
                scores = output['scores'].view(-1).tolist()
                labels = output['labels'].view(-1).tolist()

            outmsg.boxes_shape = list(output['boxes'].shape)
            outmsg.scores_shape = list(output['scores'].shape)
            outmsg.labels_shape = list(output['labels'].shape)

            outmsg.boxes = boxes
            outmsg.scores = scores
            outmsg.labels = [int(label) for label in labels]

        else:

            outmsg.type = 'segmentation'

            if output.is_cuda:
                mask = output.view(-1).cpu().tolist()
            else:
                mask = output.view(-1).tolist()

            outmsg.mask_shape = list(output.shape)

            outmsg.mask = mask

        return outmsg


    def outputcv_to_torch(self, outmsg):
        # Converts either a OutputCV32 or OutputCV64 ROS message into a pytorch tensor
        # or a dictionary of pytorch tensors (depending on if it is a segmentor or detector output).

        if outmsg.type == 'detection':

            tensor = {}
            tensor['boxes'] = torch.Tensor(outmsg.boxes).view(outmsg.boxes_shape)
            tensor['scores'] = torch.Tensor(outmsg.scores).view(outmsg.scores_shape)
            tensor['labels'] = torch.Tensor(outmsg.labels).view(outmsg.labels_shape)

        elif outmsg.type == 'segmentation':

            tensor = torch.Tensor(outmsg.mask).view(outmsg.mask_shape)

        else:

            print('"{}" is an unsupported OutputCV type'.format(outmsg.type))
            return

        return tensor, outmsg.img_resolution, outmsg.label_list

    def outputcv_to_np(self, outmsg):
        
        if outmsg.type == 'detection':

            ndarray = {}
            boxes = np.array(outmsg.boxes)
            ndarray['boxes'] = np.reshape(boxes, tuple(outmsg.boxes_shape))
            scores = np.array(outmsg.scores)
            ndarray['scores'] = np.reshape(scores, tuple(outmsg.scores_shape))
            labels = np.array(outmsg.labels)
            ndarray['labels'] = np.reshape(labels, tuple(outmsg.labels_shape))

        elif outmsg.type == 'segmentation':

            mask = np.array(outmsg.mask)
            ndarray = np.reshape(mask, tuple(outmsg.mask_shape))

        else:

            print('"{}" is an unsupported OutputCV type'.format(outmsg.type))
            return

        return ndarray, outmsg.img_resolution, outmsg.label_list