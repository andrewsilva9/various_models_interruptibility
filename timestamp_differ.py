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


def get_data_arrays():
    annots_dir = '/Users/andrewsilva/my_rel_data/'
    X = []
    Y = []
    bad_keys = ['bb_y', 'bb_x', 'gaze_timestamp', 'pos_frame', 'send_timestamp', 'bb_height', 'pos_timestamp',
                'timestamp', 'bb_width', 'objects_timestamp', 'pos_z', 'pos_x', 'pos_y', 'name', 'pose_timestamp',
                'objects']
    for filename in os.listdir(annots_dir):
        af = open(os.path.join(annots_dir, filename), 'rb')
        annotation = pickle.load(af)
        # For each person in the segmentation
        for person in annotation:
            last_value = -1
            sorted_ann = sorted(annotation[person], key = lambda x: x['timestamp'])
            for piece in sorted_ann:
                new_input_data = []
                for key, value in piece.iteritems():
                    if key in bad_keys:
                        continue
                    if value >= 1.7e300:
                        piece[key] = -5
                label = piece['value']
                if label == 0:
                    last_value = -1
                    continue
                new_input_data.append(piece['right_wrist_angle'])       # 0
                new_input_data.append(piece['right_elbow_angle'])       # 1
                new_input_data.append(piece['left_wrist_angle'])        # 2
                new_input_data.append(piece['left_elbow_angle'])        # 3
                # new_input_data.append(piece['shoulder_vec_x'])          # 4
                # new_input_data.append(piece['shoulder_vec_y'])          # 5
                new_input_data.append(piece['left_eye_angle'])          # 6
                new_input_data.append(piece['right_eye_angle'])         # 7
                # new_input_data.append(piece['eye_vec_x'])               # 8
                # new_input_data.append(piece['eye_vec_y'])               # 9
                new_input_data.append(piece['right_shoulder_angle'])    # 10
                new_input_data.append(piece['left_shoulder_angle'])     # 11
                new_input_data.append(piece['nose_vec_y'])              # 12
                new_input_data.append(piece['nose_vec_x'])              # 13
                # new_input_data.append(piece['left_arm_vec_x'])          # 14
                # new_input_data.append(piece['left_arm_vec_y'])          # 15
                # new_input_data.append(piece['right_arm_vec_x'])         # 16
                # new_input_data.append(piece['right_arm_vec_y'])         # 17
                new_input_data.append(piece['gaze'])                    # 18
                new_input_data.append(0)                                # book 19
                new_input_data.append(0)                                # bottle 20
                new_input_data.append(0)                                # bowl 21
                new_input_data.append(0)                                # cup 22
                new_input_data.append(0)                                # laptop 23
                new_input_data.append(0)                                # cell phone 24
                new_input_data.append(0)                                # blocks 25
                new_input_data.append(0)                                # tablet 26
                new_input_data.append(0)                                # unknown 27
                foi = 11 # first object index to make it easier to change stuff
                for item in piece['objects']:
                    if item == 'book':
                        new_input_data[foi] += 1
                    elif item == 'bottle':
                        new_input_data[foi+1] += 1
                    elif item == 'bowl':
                        new_input_data[foi+2] += 1
                    elif item == 'cup':
                        new_input_data[foi+3] += 1
                    elif item == 'laptop':
                        new_input_data[foi+4] += 1
                    elif item == 'cell phone':
                        new_input_data[foi+5] += 1
                    elif item == 'blocks':
                        new_input_data[foi+6] += 1
                    elif item == 'tablet':
                        new_input_data[foi+7] += 1
                    else:
                        new_input_data[foi+8] += 1
                if use_prior:
                    if np.random.rand() > dropout_val:
                        new_input_data.append(last_value)
                    else:
                        new_input_data.append(-1)

                if label <= 2:
                    label = 0
                else:
                    label = 1
                last_value = label
                X.append(new_input_data)
                Y.append(label)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y