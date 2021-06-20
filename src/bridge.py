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


def imgmsg_to_np(img_msg):
    # Since CvBride is shitty, here is a way to directly convert a compressed image
    # message to a numpy array.

    img = np.fromstring(img_msg.data, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    return img


def np_to_imgmsg(img):
    # Again, since CvBride is shitty, here is a way to directly convert a numpy array
    # to a compressed image message.

    imgmsg = CompressedImage()
    imgmsg.header.stamp = rospy.Time.now()
    imgmsg.format = "jpg"
    _, img_enc = cv2.imencode('.jpg', img)
    imgmsg.data = np.array(img_enc).tostring()

    return imgmsg



class OutputCVBridge:

    def __init__(self, bit=32):
        # The bit parameter is by default 32 bit which means that the class will
        # convert torch tensors to OutputCV32 messages in the torch_to_outputcv()
        # function. Can be changed to 64 which will do the same but for OutputCV64
        # message types instead.
        
        if bit == 32:
            self.outmsg = OutputCV32()
        elif bit == 64:
            self.outmsg = OutputCV64()
        else:
            print('{} bit message types not supported'.format(bit))
            return


    def torch_to_outputcv(self, output):
        # Takes in as input the detection output from either a box-detection or a 
        # segmentation pytorch model and converts it into an OutputCV ROS message.

        if isinstance(output, dict):
            self.outmsg.type = 'detection'

            if output['boxes'].is_cuda:
                boxes = output['boxes'].view(-1).cpu().tolist()
                scores = output['scores'].view(-1).cpu().tolist()
                labels = output['labels'].view(-1).cpu().tolist()
            else:
                boxes = output['boxes'].view(-1).tolist()
                scores = output['scores'].view(-1).tolist()
                labels = output['labels'].view(-1).tolist()

            self.outmsg.boxes_shape = list(output['boxes'].shape)
            self.outmsg.scores_shape = list(output['scores'].shape)
            self.outmsg.labels_shape = list(output['labels'].shape)

            self.outmsg.boxes = boxes
            self.outmsg.scores = scores
            self.outmsg.labels = [int(label) for label in labels]

        else:

            self.outmsg.type = 'segmentation'

            if output.is_cuda:
                mask = output.view(-1).cpu().tolist()
            else:
                mask = output.view(-1).tolist()

            self.outmsg.mask_shape = list(output.shape)

            self.outmsg.mask = mask

        return self.outmsg


    def outputcv_to_torch(self, new_outmsg):
        # Converts either a OutputCV32 or OutputCV64 ROS message into a pytorch tensor
        # or a dictionary of pytorch tensors (depending on if it is a segmentor or detector output).

        if new_outmsg.type == 'detection':

            tensor = {}
            tensor['boxes'] = torch.Tensor(new_outmsg.boxes).view(new_outmsg.boxes_shape)
            tensor['scores'] = torch.Tensor(new_outmsg.scores).view(new_outmsg.scores_shape)
            tensor['labels'] = torch.Tensor(new_outmsg.labels).view(new_outmsg.labels_shape)

        elif new_outmsg.type == 'segmentation':

            tensor = torch.Tensor(new_outmsg.mask).view(new_outmsg.mask_shape)

        else:

            print('"{}" is an unsupported OutputCV type'.format(new_outmsg.type))
            return

        return tensor




class test_pub:

    def __init__(self, bit=32):

        if bit == 32:
            msg_type = OutputCV32
        elif bit == 64:
            msg_type = OutputCV64
        else:
            print('{} bit message types not supported'.format(bit))
            return

        self.bridge = OutputCVBridge()
        self.pub = rospy.Publisher('/pub', msg_type, queue_size=1)
        self.test_ten = {}

    def publish(self):

        self.test_ten['boxes'] = torch.rand((5,4,3), dtype=torch.float32).cuda()
        self.test_ten['scores'] = torch.rand((5,4,3,90,2000), dtype=torch.float32).cuda()
        self.test_ten['labels'] = torch.Tensor(list(range(5000))).cuda()
        self.test_ten['boxes'][0,0,0] = rospy.get_time()
        outmsg = self.bridge.torch_to_outputcv(self.test_ten)
        self.pub.publish(outmsg)


class test_sub:

    def __init__(self, bit=32):

        if bit == 32:
            msg_type = OutputCV32
        elif bit == 64:
            msg_type = OutputCV64
        else:
            print('{} bit message types not supported'.format(bit))
            return

        self.bridge = OutputCVBridge()
        self.sub = rospy.Subscriber('/pub', msg_type, self.cb)
        

    def cb(self, outmsg):

        tensor = self.bridge.outputcv_to_torch(outmsg)
        if isinstance(tensor, dict):
            end = rospy.get_time()
            print(tensor['boxes'].shape)
            print(tensor['scores'].shape)
            print('Time from send to receive: {} seconds'.format(end - tensor['boxes'][0,0,0]))
        else:
            print(tensor.shape)

if __name__ == '__main__':

    # rospy.init_node('bridge_test_pub')

    # pub = test_pub()

    # while not rospy.is_shutdown():
    #     pub.publish()


    bridge = OutputCVBridge(bit=64)
    test_detect = {}
    test_detect['boxes'] = torch.rand((5,4,3), dtype=torch.float64).cuda()
    test_detect['scores'] = torch.rand((5,4,3,90,2000), dtype=torch.float64).cuda()
    test_detect['labels'] = torch.Tensor([list(range(10)), list(range(10))]).cuda()
    start = time.time()
    outmsg = bridge.torch_to_outputcv(test_detect)
    reconstructed_tensor = bridge.outputcv_to_torch(outmsg)
    end = time.time()
    # print(outmsg)
    print('Time to convert both ways: {} seconds'.format(end-start))









'''ROS way'''
# if __name__ == '__main__':

#     rospy.init_node('example')

#     detector = Detector()
#     actuator = Actuator()

#     rospy.spin()

'''Pythonic way'''
# if __name__ == '__main__':

#     rospy.init_node('example')

#     detector = Detector()
#     actuator = Actuator()

#     while not rospy.is_shutdown():

#         output = detector.detect()

#         actuator.publish_action(output)

#         rospy.sleep()