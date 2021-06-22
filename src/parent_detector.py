#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from mr_cv.msg import OutputCV32
from mr_cv.msg import OutputCV64

import cv2
import numpy as np
import torch
from time import time
import sys
import rospkg

from utils.bridge import ImgBridge, OutputCVBridge
from models.segmentor_models import COCO_Segmentor


class Detection_Publisher:
    # This is an abstract detector class
    # Each detector class will inherit some general methods from this class
    

    def __init__(self, use_topics=True, publish_img=True):

        # Must define the model in the class

        self.use_topics = use_topics

        self.model = COCO_Segmentor()

        self.img_bridge = ImgBridge()

        self.output_bridge = OutputCVBridge()

        self.img_pub = rospy.Publisher('/output_img/compressed', CompressedImage, queue_size=10)

        self.output_pub = rospy.Publisher('/output', OutputCV32, queue_size=1)

        self.img_sub = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.img_cb)

        self.start = time()


    def img_cb(self, img_msg):
        # Subscribes to the compressed image message and converts it to a numpy array,
        # then performs a detection on it, converts it back into a compressed
        # image message and publishes it.

        sys.stdout.write("\033[F") #back to previous line.
        sys.stdout.write("\033[K") #clear line.

        img = self.img_bridge.imgmsg_to_np(img_msg)

        img_detected = self.detect(img)
        
        imgmsg_detected = self.img_bridge.np_to_imgmsg(img_detected)

        if self.use_topics:

            self.img_pub.publish(imgmsg_detected)

        now = time()

        fps = 1/(now-self.start)

        self.start = now

        rospy.sleep(0.001)

        print('{} FPS'.format(round(fps, 2)))



    def max_scoring_preds(self, boxes, labels, scores, labels_lst, score_threshold=0):

        max_boxes = {}

        for label in labels_lst:

            inds = np.where(labels == label)[0]
            label_boxes = boxes[inds]
            label_scores = scores[inds]
            max_box = []

            if inds.size > 0:
                max_ind = np.argmax(label_scores)
                if label_scores[max_ind] > score_threshold:
                    max_box = label_boxes[max_ind]

            max_boxes[label] = max_box

        return max_boxes


    def draw_box(self, img, box, color, label=''):

        cv2.rectangle(
                        img,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])), 
                        color, 4
                    )

        if label != '':
            self.label_box(img, box, label, color)

    
    def label_box(self, img, box, label, color):

        
  
        org = (int(box[0]+10), int(box[1]+40))

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 2

        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[0]+(20*len(label))), int(box[1]+50)),  (0,0,0), -1)

        image = cv2.putText(img, label, org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)


    def decode_segmap(self, image, orig_image, nc=21):
        # Source for this code: https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/

        label_colors = np.array([  # 0=background
                (0,0,0),
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 2]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 0]
        bgr = np.stack([b, g, r], axis=2)
        bgr = np.where(bgr==[0,0,0], orig_image, bgr)
        return bgr


    def detect(self, img):

        output = self.model.forward(img).detach().cpu().numpy()
        output_img = self.decode_segmap(output, img)
        # boxes = output['boxes'].detach().cpu().numpy()
        # scores = output['scores'].detach().cpu().numpy()
        # labels = output['labels'].detach().cpu().numpy()
        # for i, box in enumerate(boxes):
        #     if scores[i] > 0.5:
        #         color = (0,255,0)
        #         self.draw_box(img, box, color, label=self.model.label_dict[labels[i]])
        
        return output_img



if __name__ == '__main__':

    rospy.init_node('detector')

    detector = Detection_Publisher()

    rospy.spin()