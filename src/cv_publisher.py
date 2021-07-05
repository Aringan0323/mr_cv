#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from mr_cv.msg import OutputCV

import cv2
import numpy as np
import torch
import torchvision as tv
from time import time
import sys
import rospkg

from utils.bridge import ImgBridge, OutputCVBridge
from models.segmentor_models import COCO_Segmentor_Fast, COCO_Segmentor_Accurate
from models.detector_models import COCO_Detector_Fast, COCO_Detector_Accurate, PersonFace_Detector



class CV_Publisher:
    # The user can input a keyword that corresponds to the type
    # of model that they would like to use, and then this class 
    # will publish an OutputCV topic as well as a marked-up image
    # of the detection when rospy.spin() is called.

    # The user can also choose not to publish topics and instead
    # simply receive the raw model output from the detect() function
    # if they specify use_topics=False.
    

    def __init__(self, model_keyword, use_topics=True, visualization_threshold=0.6):

        model_dict = {'coco_segmentor_fast':COCO_Segmentor_Fast, 
                        'coco_segmentor_accurate':COCO_Segmentor_Accurate, 
                        'coco_detector_fast':COCO_Detector_Fast,
                        'coco_detector_accurate':COCO_Detector_Accurate,
                        'personface_detector':PersonFace_Detector}

        self.model = model_dict[model_keyword]()

        self.use_topics = use_topics

        self.visualization_threshold = visualization_threshold

        self.img_bridge = ImgBridge()

        if self.use_topics:

            self.output_bridge = OutputCVBridge()

            self.img_pub = rospy.Publisher('/output_img/compressed', CompressedImage, queue_size=10)

            self.output_pub = rospy.Publisher('/output', OutputCV, queue_size=1)

            self.sigmoid = torch.nn.Sigmoid()

        self.img = None

        self.img_sub = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.img_cb)

        self.start = time()


    def img_cb(self, img_msg):
        # Subscribes to the compressed image message and converts it to a numpy array,
        # then performs a detection on it, converts it back into a compressed
        # image message and publishes it.

        self.img = self.img_bridge.imgmsg_to_np(img_msg)

        if self.use_topics:

            output = self.detect()
            self.visualize_output(output)

            img_resolution = list(self.img.shape[0:2])

            outmsg = self.output_bridge.torch_to_outputcv(output, img_resolution, self.model.label_list)
            self.output_pub.publish(outmsg)

            imgmsg_detected = self.img_bridge.np_to_imgmsg(self.img)
            self.img_pub.publish(imgmsg_detected)

            rospy.sleep(0.01)

            sys.stdout.write("\033[F") #back to previous line.
            sys.stdout.write("\033[K") #clear line.

            now = time()

            fps = 1/(now-self.start)

            self.start = now

            print('{} FPS'.format(round(fps, 2)))


    def bool_mask(self, masks):

        bool_masks = torch.ones(masks.shape, dtype=torch.bool).cuda()

        for i in range(masks.shape[0]):
            bool_masks[i] = (masks.argmax(0) == i)
        
        return bool_masks


    def detect(self):

        if self.img is not None:
            output = self.model.forward(self.img)
        else:
            output = None

        return output


    def visualize_output(self, output):
        if output is not None:

            torch_img = torch.from_numpy(self.img.transpose((2, 0, 1)))

            if isinstance(output, dict):

                boxes = output['boxes']
                scores = output['scores']
                labels = output['labels']

                confident_boxes = boxes[scores >= self.visualization_threshold]
                confident_labels = labels[scores >= self.visualization_threshold]
                confident_scores = scores[scores >= self.visualization_threshold]

                box_labels = [self.model.label_dict[int(id)] + ": " + str(round(float(confident_scores[i])*100)) + "%" for i, id in enumerate(confident_labels)]

                self.img = tv.utils.draw_bounding_boxes(torch_img, 
                                                        confident_boxes, 
                                                        labels=box_labels,
                                                        colors=[(0,255,0)]*len(box_labels) 
                                                        ).numpy().transpose((1, 2, 0))

            else:
                bool_masks = self.bool_mask(output)
                self.img = tv.utils.draw_segmentation_masks(torch_img, bool_masks, alpha=.8).numpy().transpose((1, 2, 0))
        else:
            pass





if __name__ == '__main__':


    rospy.init_node('detector')

    model = str(rospy.get_param('~model'))

    detector = CV_Publisher('coco_detector_accurate')
    
    rospy.spin()