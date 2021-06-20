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
        


    def torch_to_outputcv(self, output):
        # Takes in as input the detection output from either a box-detection or a 
        # segmentation pytorch model and converts it into an OutputCV ROS message.

        if self.bit == 32:
            outmsg = OutputCV32()
        elif self.bit == 64:
            outmsg = OutputCV64()
        else:
            print('{} bit message types not supported'.format(bit))
            return

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
        

    def publish(self, now):
        test_ten = {}
        test_ten['boxes'] = torch.rand((1)).cuda()
        test_ten['scores'] = torch.rand((1)).cuda()
        test_ten['labels'] = torch.ones((1)).cuda() * now
        outmsg = self.bridge.torch_to_outputcv(test_ten)
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
        print(float(tensor['labels']))
        # if isinstance(tensor, dict):
        #     end = time.time()
        #     delta = end - float(tensor['boxes'][0,0,0])
        #     print(end)
        #     print(float(tensor['boxes'][0,0,0]))
        #     print(delta)
        #     print('\n\n')
        # else:
        #     print(tensor.shape)

if __name__ == '__main__':

    rospy.init_node('bridge_test_pub')

    pub = test_pub()

    while not rospy.is_shutdown():
        pub.publish(time.time())


    # bridge = OutputCVBridge(bit=64)
    # test_detect = {}
    # test_detect['boxes'] = torch.rand((1,400,4), dtype=torch.float64).cuda()
    # test_detect['scores'] = torch.rand((1,400), dtype=torch.float64).cuda()
    # test_detect['labels'] = torch.Tensor((1,400)).cuda()

    # start = time.time()
    # outmsg = bridge.torch_to_outputcv(test_detect['boxes'])
    # reconstructed_tensor = bridge.outputcv_to_torch(outmsg)
    # end = time.time()
    # # print(outmsg)
    # print('Time to convert both ways: {} seconds'.format(end-start))









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