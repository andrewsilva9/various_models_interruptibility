#!/usr/bin/env python
# This script takes in segmentations from /collections, pre-eyisting annotations from /my_pickle_out, and copies the
# values to corresponding segments in a new directory /copied_annotations
# Andrew Silva 05/30/17
import pickle
import json
import os
import math
import sys
import numpy as np
import cv2
import collections


def get_data_arrays():
    annots_dir = '/home/asilva/my_rel_data/'
    Y = {}
    bad_keys = ['bb_y', 'bb_x', 'gaze_timestamp', 'pos_frame', 'send_timestamp', 'bb_height', 'pos_timestamp',
                'timestamp', 'bb_width', 'objects_timestamp', 'pos_z', 'pos_x', 'pos_y', 'name', 'pose_timestamp',
                'objects']
    for filename in os.listdir(annots_dir):
        af = open(os.path.join(annots_dir, filename), 'rb')
        annotation = pickle.load(af)
        # For each person in the segmentation
        for person in annotation:
            timestamp_person_dict = collections.OrderedDict()
            sorted_ann = sorted(annotation[person], key = lambda x: x['timestamp'])
            for piece in sorted_ann:
                label = piece['value']
                # For unknown interruptibility / missing people, just move on
                if label == 0:
                    continue

                # Binarize our labels
                if label <= 2:
                    label = 0
                else:
                    label = 1
                timestamp_person_dict[piece['timestamp']] = label
            dict_name = str(filename).split("annotation-")[1]
            dict_name = dict_name.split(".pickle")[0]
            Y[dict_name + " " + str(person)] = timestamp_person_dict
    return Y

print get_data_arrays()