#!/usr/bin/env python3

import rospy
import time

from nodes import Selector, Sequencer, Action, Conditional


'''
This file is an example of how a behavior tree can be constructed in a python
script with the different types of nodes in the nodes.py file. This file shows a
simple behavior tree construction where we have a blackboard with "environment states"
and nodes which interact with and view that environment.
'''

def door_cb(blackboard):
    
    return blackboard['open']

def key_cb(blackboard):

    return blackboard['key']

def person_cb(blackboard):

    return blackboard['person nearby']

def crowbar_cb(blackboard):

    return blackboard['crowbar']

def door_type_cb(blackboard):

    door_type = blackboard['door type']

    if door_type == 'thin':
        return True
    else:
        print('door is too thick')
        return False


class OpenWithKey(Action):

    def tick(self, blackboard):

        if blackboard['door health'] > 10:
            print('Door is barracaded, key did not work')
            return 'failure'
        else:
            blackboard['open'] = True

            print('Opened door with key')

            return 'success'


class PersonOpen(Action):

    def tick(self, blackboard):

        if blackboard['person nice']:

            print('Had person open door')
            blackboard['open'] = True
            return 'success'
        else:
            print('Person did not open door')
            return 'failure'


class BreakDoor(Action):
    
    def tick(self, blackboard):

        blackboard['door health'] -= 1

        print('Hit door')

        if blackboard['door health'] > 0:
            return 'running'
        else:
            print('Door broken down')
            blackboard['open'] = True
            return 'success'








if __name__ == '__main__':

    blackboard = {
        'open':False,
        'key':True,
        'door jammed':True,
        'person nearby':True,
        'person nice':False,
        'crowbar':True,
        'door type':'thin',
        'door health':70
    }


    door_open = Conditional(door_cb)
    has_key = Conditional(key_cb)
    person_nearby = Conditional(person_cb)
    has_crowbar = Conditional(crowbar_cb)
    thin_door = Conditional(door_type_cb)

    open_door_key = OpenWithKey()
    person_open_door = PersonOpen()
    break_door = BreakDoor()

    key_seq = Sequencer([has_key, open_door_key])
    person_seq = Sequencer([person_nearby, person_open_door])
    break_seq = Sequencer([has_crowbar, thin_door, break_door])

    open_sel = Selector([door_open, key_seq, person_seq, break_seq])

    status = 'running'
    i = 1
    start = time.time()
    while status == 'running':
        print('\nTick {}:\n'.format(i))
        i += 1
        status = open_sel.tick(blackboard)
    print(time.time() - start)
    print(status)


    