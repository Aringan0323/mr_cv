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
from models.detector_models import COCO_Detector, PersonFace_Detector


class CV_Publisher:
    # This is an abstract detector class
    # Each detector class will inherit some general methods from this class
    

    def __init__(self, model_keyword, use_topics=True):

        model_dict = {'coco_segmentor':COCO_Segmentor, 'coco_detector':COCO_Detector, 'personface_detector':PersonFace_Detector}

        # Must define the model in the class

        self.use_topics = use_topics

        self.model = model_dict[model_keyword]()

        self.img_bridge = ImgBridge()

        if self.use_topics:

            self.output_bridge = OutputCVBridge()

            self.img_pub = rospy.Publisher('/output_img/compressed', CompressedImage, queue_size=10)

            self.output_pub = rospy.Publisher('/output', OutputCV32, queue_size=1)

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

            outmsg = self.output_bridge.torch_to_outputcv(output)
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


    def draw_box(self, box, label=''):

        cv2.rectangle(
                        self.img,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])), 
                        (0,255,0), 4
                    )

        if label != '':
            self.label_box(box, label)

    
    def label_box(self, box, label):

        org = (int(box[0]+10), int(box[1]+40))

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 2

        cv2.rectangle(self.img, (int(box[0]), int(box[1])), (int(box[0]+(20*len(label))), int(box[1]+50)),  (0,0,0), -1)

        image = cv2.putText(self.img, label, org, font, 
                        fontScale, (255,255,255), thickness, cv2.LINE_AA)


    def decode_segmap(self, output, nc=21):
        # Source for this code: https://learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/
        output = output.cpu().numpy()
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
        r = np.zeros_like(output).astype(np.uint8)
        g = np.zeros_like(output).astype(np.uint8)
        b = np.zeros_like(output).astype(np.uint8)
        for l in range(0, nc):
            idx = output == l
            r[idx] = label_colors[l, 2]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 0]
        bgr = np.stack([b, g, r], axis=2)
        bgr = np.where(bgr==[0,0,0], self.img, bgr)
        self.img = bgr


    def detect(self):

        if self.img is not None:
            output = self.model.forward(self.img)
        else:
            output = None

        return output


    def visualize_output(self, output):
        if output is not None:
            if isinstance(output, dict):

                boxes = output['boxes']
                scores = output['scores']
                labels = output['labels']
                for i in range(boxes.shape[0]):
                    if scores[i] >= 0.8:
                        label_id = labels[i]
                        label = self.model.label_dict[int(label_id)]
                        self.draw_box(boxes[i], label=label)
            else:

                self.decode_segmap(output)
        else:
            pass





if __name__ == '__main__':

    rospy.init_node('detector')

    detector = CV_Publisher('personface_detector')
    
    rospy.spin()