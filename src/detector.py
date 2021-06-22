#!/usr/bin/env python3

import rospy
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage

import cv2
import numpy as np
import torch
from time import time
import sys

from .models import detector_models

import .bridge


class detector:
    # This is an abstract detector class
    # Each detector class will inherit some general methods from this class
    

    def __init__(self):

        # Must define the model in the class

        self.img_sub = rospy.Subscriber('/raspicam_node/image/compressed', CompressedImage, self.img_cb)

        self.img_pub = rospy.Publisher('/output/compressed', CompressedImage, queue_size=1)

        self.start = time()


    def img_cb(self, img_msg):
        # Subscribes to the compressed image message and converts it to a numpy array,
        # then performs a detection on it, converts it back into a compressed
        # image message and publishes it.

        sys.stdout.write("\033[F") #back to previous line.
        sys.stdout.write("\033[K") #clear line.

        img = bridge.imgmsg_to_np(img_msg)

        img_detected = self.detect(img)
        
        imgmsg_detected = bridge.np_to_imgmsg(img_detected)

        self.img_pub.publish(imgmsg_detected)

        now = time()

        fps = 1/(now-self.start)

        self.start = now

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


    def detect(self, img):
        
        return img


class person_face_detector(detector):


    def __init__(self):
        
        self.model = face_person_mobilenet()

        print('Loaded model')

        self.person_pub = rospy.Publisher('/person_centroid', Int32MultiArray, queue_size=1)

        self.face_pub = rospy.Publisher('/face_detected', Bool, queue_size=1)

        super().__init__()


    def publish_data(self, person_box, face_box):

        cX, cY = -1, -1
        face_detected = False

        if len(person_box) > 0:
            cX = int(person_box[0] + ((person_box[2]-person_box[0])/2))
            cY = int(person_box[1] + ((person_box[3]-person_box[1])/2))
        if len(face_box) > 0:
            face_detected = True
        
        person_msg = Int32MultiArray(data=[cX, cY])
        face_msg = Bool(data=face_detected)
        
        self.person_pub.publish(person_msg)
        self.face_pub.publish(face_msg)
        

    def detect(self, img):
        boxes, labels, scores = self.model.simple_detection(img)
        
        max_boxes = self.max_scoring_preds(boxes, labels, scores, [1,2], score_threshold=0.9)
        person_box, face_box = max_boxes[1], max_boxes[2]
        self.publish_data(person_box, face_box)
        
        if len(person_box) > 0:
            self.draw_box(img, person_box, (255,0,0))
        if len(face_box) > 0:
            self.draw_box(img, face_box, (0,0,255))

        return img


class coco_detector(detector):


    def __init__(self):
        
        self.model = coco_mobilenet()

        self.label_dict = self.model.category_map

        print('Loaded model')

        super().__init__()


    def detect(self, img):

        boxes, labels, scores = self.model.simple_detection(img)

        label_lst = list(range(1,92))

        max_boxes = self.max_scoring_preds(boxes, labels, scores, label_lst, score_threshold=.7)

        for label in label_lst:
            box = max_boxes[label]

            if len(box) > 0:

                name = self.label_dict[label]

                self.draw_box(img, box, (0,255,0), label=name)

        return img


class coco_segmentor(detector):

    def __init__(self):

        self.model = coco_segmentation_mobilenet()

        super().__init__()


    def detect(self, img):

        img = self.model.simple_segmentation(img)

        return img


if __name__ == '__main__':

    rospy.init_node('detector')

    detector = coco_segmentor()

    rospy.spin()
    