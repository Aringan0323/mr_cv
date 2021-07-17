#!/usr/bin/env python3

import numpy as np
import json
import graphviz
import sys

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage

from nodes import Conditional, Action, Update, Sequencer, Selector

from action_nodes.basic_movement import LinearStatic, LinearDynamic, AngularStatic, AngularDynamic, LinearAngularStatic, LinearAngularDynamic, Stop

from update_nodes.basic_updates import FlipBoolVar, IncrementVar, OffsetVar
from update_nodes.movement_control_updates import LinearPID, AngularPID
from update_nodes.cv_updates import FastDetector, ItemBearingErr
from update_nodes.scan_updates import CalcNearestWallAngle, CalcNearestDist, CalcAvgFrontDist

from conditional_nodes.scan_conditionals import WallAhead, ClearAhead
from conditional_nodes.basic_conditionals import BoolVar, BoolVarNot

from ros_behavior_tree import ROSBehaviorTree




master_dict = {
    
    "Conditional":Conditional, "Action":Action, "Update":Update, "Sequencer":Sequencer, "Selector":Selector, 
    "LinearStatic":LinearStatic, "LinearDynamic":LinearDynamic, "AngularStatic":AngularStatic, "AngularDynamic":AngularDynamic,
    "LinearAngularStatic":LinearAngularStatic, "LinearAngularDynamic":LinearAngularDynamic, "Stop": Stop,
    "FlipBoolVar":FlipBoolVar, "IncrementVar":IncrementVar, "OffsetVar":OffsetVar,
    "LinearPID":LinearPID, "AngularPID":AngularPID,
    "FastDetector":FastDetector, "ItemBearingErr":ItemBearingErr,
    "CalcNearestWallAngle":CalcNearestWallAngle, "CalcNearestDist":CalcNearestDist, "CalcAvgFrontDist":CalcAvgFrontDist,
    "WallAhead":WallAhead, "ClearAhead":ClearAhead,
    "BoolVar":BoolVar, "BoolVarNot":BoolVarNot
}



class TreeGrapher:

    def __init__(self, path, comment=""):

        with open(path) as f:
            self.tree_dict = json.load(f)

        self.dot = graphviz.Digraph(comment='Behavior Tree')

        self.node_label = 0

        self.link_nodes(self.tree_dict)


    def draw_graph(self):
        
        # self.dot.format = 'png'
        # self.dot.render()
        # print(self.dot)
        self.dot.view()


    def link_nodes(self, node, parent_label=None):

        label_string = node['name'] + "\ntype: " + node['type']

        string_node_label = str(self.node_label)

        if node['type'] == 'Selector':
            shape = "box"
        elif node['type'] == "Sequencer":
            shape = "cds"
        else:
            shape = "oval"

        self.dot.node(string_node_label, label_string, shape=shape)

        if 'children' in node:

            for child in node['children']:

                self.node_label += 1

                child_label = self.link_nodes(child, parent_label=string_node_label)

                self.dot.edge(string_node_label, child_label)

        if 'blackboard' in node:
            blackboard_string = 'BLACKBOARD\n\n'
            for key in node['blackboard']:
                blackboard_string += key + '  :  ' + str(node['blackboard'][key]) + '\n'
            self.dot.node('Blackboard', blackboard_string, shape='rectangle')

        return string_node_label




# class TreeBuilder

if __name__ == '__main__':

    tg = TreeGrapher('tree_jsons/test.json')
    tg.draw_graph()