#!/usr/bin/env python3

import rospy
import numpy as np
import json
import graphviz
import sys

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage

from nodes import Conditional, Action, Update, Sequencer, Selector, Multitasker

from action_nodes.basic_movement import LinearStatic, LinearDynamic, AngularStatic, AngularDynamic, LinearAngularStatic, LinearAngularDynamic, Stop

from update_nodes.basic_updates import FlipBoolVar, IncrementVar, OffsetVar
from update_nodes.movement_control_updates import LinearPID, AngularPID
from update_nodes.cv_updates import FastDetector, ItemBearingErr
from update_nodes.scan_updates import CalcNearestWallAngle, CalcNearestDist, CalcAvgFrontDist

from conditional_nodes.scan_conditionals import WallAhead, ClearAhead
from conditional_nodes.basic_conditionals import BoolVar, BoolVarNot

from ros_behavior_tree import ROSBehaviorTree




master_node_dict = {
    
    "Conditional":Conditional, "Action":Action, "Update":Update, "Sequencer":Sequencer, "Selector":Selector, "Multitasker":Multitasker,
    "LinearStatic":LinearStatic, "LinearDynamic":LinearDynamic, "AngularStatic":AngularStatic, "AngularDynamic":AngularDynamic,
    "LinearAngularStatic":LinearAngularStatic, "LinearAngularDynamic":LinearAngularDynamic, "Stop": Stop,
    "FlipBoolVar":FlipBoolVar, "IncrementVar":IncrementVar, "OffsetVar":OffsetVar,
    "LinearPID":LinearPID, "AngularPID":AngularPID,
    "FastDetector":FastDetector, "ItemBearingErr":ItemBearingErr,
    "CalcNearestWallAngle":CalcNearestWallAngle, "CalcNearestDist":CalcNearestDist, "CalcAvgFrontDist":CalcAvgFrontDist,
    "WallAhead":WallAhead, "ClearAhead":ClearAhead,
    "BoolVar":BoolVar, "BoolVarNot":BoolVarNot
}

master_msg_dict = {

    "Twist":Twist, "LaserScan":LaserScan, "CompressedImage":CompressedImage
}



'''

RULES FOR THE JSON FORMATTING:

    NODES:

        For each node in a tree, you must provide a "name" parameter and a "type" parameter.

            The "name" field is a string that will be displayed in the Graphviz tree along with the "type" for each node. Each node in a tree must have
            a unique "name" in order for the tree to be displayed properly, but other than that the "name" of a node is arbitrary.

            The "type" parameter is used to specify which type of node you are instantiating. You must use one of the currently available
            node types listed above in the master_node_dict.

        When you are declaring a parent node, you will have a "children" parameter that will ask for a list of other nodes. You must provide
        a list of newly specified nodes in the same format as you would provide information for a regular node. This will give your .json file
        a nested structure.

    REFERENCES:

        You may pass in a reference to another json file node/tree structure as a child of another node. To do this, when declaring the node you must
        pass in an argument "ref" and assign it to the path of the referenced file relative to the interpreter.

    BLACKBOARD:

        You will need to provide a blackboard with the necessary variables to keep track of inside of your .json file. You will put this blackboard in
        as a parameter of the parent node and name it "blackboard".

        There are two types of blackboard variables that can be used in the blackboard.

            The "generic" variables which can be any kind of object or primitive data type supported by python. These types can have any name. They can be
            specified to initially have a null value, or start with a value of a data type supported by json.

            The ROS message variables will have the names of the topic which they are published to. Their name must start with a "/" or they will not be recognized
            as a ROS message and a subscriber will not be instantiated for them. They must initially have a value which is a key for one of the ROS message types specified
            above in the master_msg_dict.

    EXAMPLE:

        {
            "name":"parent",
            "type":"Selector",
            "children":[
                {
                    "name":"child1",
                    "type":"SomeConditionalNode",
                    "some_param1":"foo"
                },
                {
                    "name":"child2",
                    "type":"SomeActionNode",
                    "random_param1":"bar"
                },
                {
                    "ref":"path/to/other/node.json"
                }
            ],
            "blackboard":{
                "/scan":"LaserScan",
                "some_var":null
            }
        }

'''
class TreeBuilder:


    def __init__(self, path, comment=""):

        with open(path) as f:
            self.tree_dict = json.load(f)

        self.dot = graphviz.Digraph(format='pdf', comment='Behavior Tree')

        self.blackboard = None


    def build_tree(self):
        '''
        The recursive function attach_node() is called on the root of the tree, then the 
        ROS behavior tree root and the blackboard are returned.
        '''
        root = self.attach_node(self.tree_dict) 

        return root, self.blackboard  


    def attach_node(self, node):

        parameters = []

        specials = ['name', 'type', 'blackboard']

        for parameter in node: # Each parameter provided in the json is interpreted and used to initialize the node

            if parameter == 'children': # Initializes all children recursively and appends them to a list which is then
                                        # passed as another parameter in the node
                children = []

                for child in node['children']:
                    if 'ref' in child: # Handles the case where the child is a reference to another json file
                        with open(child['ref']) as f:
                            child = json.load(f)
                    children.append(self.attach_node(child))
                
                parameters.append(children)
       
            elif parameter not in specials:
                
                parameters.append(node[parameter])

        if 'blackboard' in node: # If the blackboard is passed as a parameter it is converted into the compatible list format

            for var in node['blackboard']:
                
                if var[0] == '/':
                    print(var)
                    node['blackboard'][var] = master_msg_dict[node['blackboard'][var]]

            self.blackboard = node['blackboard']   

        return master_node_dict[node['type']](*parameters)


    def draw_tree(self):
        '''
        The recursive function link_nodes() is called on the root of the tree, then the 
        the graph is drawn and a pdf is created using a Digraph object from GraphViz.
        '''

        self.link_nodes(self.tree_dict)

        self.dot.view()


    def link_nodes(self, node, parent_label=None):

        label_string = node['name'] + "\ntype: " + node['type']

        node_label = node['name']

        if node['type'] == 'Selector': # Changes the shape of the node depending on the type
            shape = "box"
        elif node['type'] == "Sequencer":
            shape = "cds"
        else:
            shape = "oval"

        self.dot.node(node_label, label_string, shape=shape)

        if 'children' in node: # Recursively creates all of the Graphviz children nodes

            for child in node['children']:

                if 'ref' in child:
                    with open(child['ref']) as f:
                        child = json.load(f)

                child_label = self.link_nodes(child, parent_label=node_label)

                self.dot.edge(node_label, child_label)

        if 'blackboard' in node: # Blackboard is visualized as being passed into the node it is initialized in
            blackboard_string = 'BLACKBOARD\n\n'
            for key in node['blackboard']:
                blackboard_string += key + '  :  ' + str(node['blackboard'][key]) + '\n'
            self.dot.node('Blackboard', blackboard_string, shape='rectangle')
            self.dot.edge('Blackboard', node_label)

        return node_label



if __name__ == '__main__':

    rospy.init_node('person_follower')


    tg = TreeBuilder('tree_jsons/item_follower/item_follower.json')
    tg.draw_tree()
    node, blackboard = tg.build_tree()


    tree = ROSBehaviorTree(node, blackboard)
    rospy.spin()