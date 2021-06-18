#!/usr/bin/env python3


import rospy
from detector import person_face_detector
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Bool
import numpy as np



class follower:

    def __init__(self):

        self.face_detected = False

        self.person_centroid = [-1,-1]

        self.ranges = [10]*360

        self.centroid_sub = rospy.Subscriber('/person_centroid', Int32MultiArray, self.centroid_cb)

        self.person_sub = rospy.Subscriber('/face_detected', Bool, self.face_cb)

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_cb)


        self.twist = Twist()

        self.twist.linear.x = 0
        self.twist.angular.z = 0

        self.cmd_vel_pub.publish(self.twist)

        
    def centroid_cb(self, msg):

        self.person_centroid = msg.data


    def face_cb(self, msg):

        self.face_detected = msg.data

        if self.face_detected:
            print("I'm frozen!!!")


    def scan_cb(self, msg):

        self.ranges = msg.ranges

        self.follow()

    def centroid_processor(self, centroids):
        h, w, d = [1080,1920,0]
        # print("Person location: ", centroids)
        err = centroids[0] - w/2
        err = -float(err) / 1200

        return err

    def follow(self):

        if self.person_centroid == [-1,-1]:
            print('No person detected!')
            self.twist.linear.x = 0
            self.twist.angular.z = 0
        else:
            print('Person detected!')
            self.twist.linear.x = 0
            self.twist.angular.z = 0
            if not self.face_detected:
                err = self.centroid_processor(self.person_centroid)

                print('Following!')
                self.twist.angular.z = err
            else:
                print("I'm frozen!")
                rospy.sleep(1)

        
        
        self.cmd_vel_pub.publish(self.twist)
        self.twist.linear.x = 0
        self.twist.angular.z = 0

            



if __name__ == '__main__':

    rospy.init_node('follower')

    detector = person_face_detector()

    follower = follower()

    rospy.spin()