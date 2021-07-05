#!/usr/bin/env python3

import rospy
import numpy as np
import torch


class ItemFilter:

    def __init__(self, item, threshold):
        
        self.item = item
        self.threshold = threshold


    def filter_output(self, output, label_list):

        if isinstance(output, dict):
            output = self.filter_boxes(output, label_list)
        else:
            output = self.filter_mask(output)

        return output


    def filter_boxes(self, output, label_list):

        item_id = label_list.index(self.item)

        # Selects boxes, scores and labels where the label matches the specified item
        filtered_boxes = output['boxes'][torch.where(output['labels'] == item_id)]
        filtered_scores =  output['scores'][torch.where(output['labels'] == item_id)]
        filtered_labels = output['labels'][torch.where(output['labels'] == item_id)]

        # Filters out outputs that score lower than the specified threshold
        if self.threshold == 1 and torch.numel(filtered_boxes) > 0:
            max_ind = torch.argmax(filtered_scores)
            filtered_boxes = torch.unsqueeze(filtered_boxes[max_ind], 0)
            filtered_labels = torch.unsqueeze(filtered_labels[max_ind], 0)
            filtered_scores = torch.unsqueeze(filtered_scores[max_ind], 0)

        elif self.threshold > 0:

            filtered_boxes = filtered_boxes[torch.where(filtered_scores >= self.threshold)]
            filtered_labels = filtered_labels[torch.where(filtered_scores >= self.threshold)]
            filtered_scores = filtered_scores[torch.where(filtered_scores >= self.threshold)]

        else:
            filtered_boxes = torch.Tensor([])
            filtered_labels = torch.Tensor([])
            filtered_scores = torch.Tensor([])

        return {'boxes':filtered_boxes, 'scores':filtered_scores, 'labels':filtered_labels}


    def filter_mask(self, mask):

        item_id = label_list.index(self.item)

        # Selects mask of the specified item
        filtered_mask = filtered_mask[item_id]

        # Sets all pixels of mask that score lower than specified threshold to 0
        if self.threshold > 0:

            filtered_mask = torch.where(filtered_mask >= self.threshold, filtered_mask, torch.zeros_like(filtered_mask))

        return torch.unsqueeze(filtered_mask,0)