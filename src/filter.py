#!/usr/bin/env python3

import rospy

from filters.item_filter import ItemFilter
from mr_cv.msg import OutputCV
from utils.bridge import OutputCVBridge


class Filter:

    def __init__(self, filter_type, *args):
        
        filter_dict = {'item_filter':ItemFilter}

        self.filter = filter_dict[filter_type](*args)

        self.bridge = OutputCVBridge()
        self.sub = rospy.Subscriber('/output', OutputCV, self.cb)
        self.pub = rospy.Publisher('/filtered_output', OutputCV, queue_size=1)

    def cb(self, msg):

        output, img_resolution, label_list = self.bridge.outputcv_to_torch(msg)

        filtered_output = self.filter.filter_output(output, label_list)

        outmsg = self.bridge.torch_to_outputcv(filtered_output, img_resolution, label_list)

        self.pub.publish(outmsg)


if __name__ == '__main__':

    rospy.init_node('filter')

    item = str(rospy.get_param('~item'))

    filter = Filter('item_filter', item, 1)

    rospy.spin()

