#!/usr/bin/env python
# This script takes in segmentations from /collections, pre-eyisting annotations from /my_pickle_out, and copies the
# values to corresponding segments in a new directory /copied_annotations
# Andrew Silva 05/30/17
import pickle
import json
import os
import math
import sys
import itertools
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.externals import joblib


blank_image = np.ones((1080, 1920, 3), np.uint8) * 255


def non_shuffling_train_test_split(X, y, test_size=0.2):
    i = int((1 - test_size) * X.shape[0]) + 1
    X_train, X_test = np.split(X, [i])
    y_train, y_test = np.split(y, [i])
    return X_train, X_test, y_train, y_test


sdir = os.path.join(os.getcwd(), 'collections')
adir = os.path.join(os.getcwd(), 'sid_pickle_out')
aoutdir = os.path.join(os.getcwd(), 'copied_annotations')

argmax = lambda x: max(x.iteritems(), key=lambda y: y[1])[0]
argmin = lambda x: min(x.iteritems(), key=lambda y: y[1])[0]


def find_closest_steps(longer_list, short_list):
    target = short_list[0]['timestamp']
    # print short_list[0]
    closest_step = 999
    best_match = []
    match_index = -1
    for index, step in enumerate(longer_list):
        if abs(step['timestamp'] - target) < closest_step:
            closest_step = abs(step['timestamp'] - target)
            best_match = step
            match_index = index
    return best_match, short_list[0], match_index


def distance_between(person1, person2):
    z_diff = (person1['pos_z'] - person2['pos_z']) ** 2
    y_diff = (person1['pos_y'] - person2['pos_y']) ** 2
    x_diff = (person1['pos_x'] - person2['pos_x']) ** 2 / 10
    return math.sqrt(z_diff+y_diff+x_diff)


def time_between(segment, annotation):
    timestep1 = segment['timestamp']
    timestep2 = annotation['timestamp']
    return abs(timestep1 - timestep2)


def annotation_printer(annotation):
    for index, person in enumerate(annotation):
        print index, person
        print index, ':', person, 'is', len(annotation[person]), 'timesteps long'
        print annotation[person]
    print ''


def angle_between(vec1, vec2):
    if vec1[0] >= 1.7e300 or vec1[1] >= 1.7e300 or vec2[0] >= 1.7e300 or vec2[1] >= 1.7e300:
        return 1.7e300
    v1norm = np.linalg.norm(vec1)
    v2norm = np.linalg.norm(vec2)
    if v1norm == 0 or v2norm == 0:
        return 1.7e3000
    vec1 = vec1 / v1norm
    vec2 = vec2 / v2norm
    return np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))*180/np.pi


def pt_dist(pt1, pt2):
    if pt1[0] >= 1.7e300 or pt1[1] >= 1.7e300 or pt2[0] >= 1.7e300 or pt2[1] >= 1.7e300:
        return 1.7e3000

    z_diff = (pt1[0] - pt2[0]) ** 2
    o_diff = (pt1[1] - pt2[1]) ** 2
    return math.sqrt(z_diff+o_diff)


def change_to_relations():
    annots_dir = '/Users/andrewsilva/my_annots_trimmed'
    # new_annots_dir = '/home/asilva/my_rel_data'
    avg_l_s, avg_r_s, avg_l_w, avg_r_w, avg_l_e, avg_r_e, avg_l_ey, avg_r_ey, avg_no, avg_ne = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    rsc, lsc, rec, lec, rwc, lwc, reyc, leyc, noc, nec = 0,0,0,0,0,0,0,0,0,0
    for filename in os.listdir(annots_dir):
        af = open(os.path.join(annots_dir, filename), 'rb')
        annotation = pickle.load(af)
        new_annotation = {}
        # For each person in the segmentation

        for person in annotation:
            last_timestep = {}
            sorted_ann = sorted(annotation[person], key = lambda x: x['timestamp'])
            for piece in sorted_ann:
                # Want distance for eye_vec
                if last_timestep == {}:
                    last_timestep = piece
                    continue
                if time_between(last_timestep, piece) > 1:
                    last_timestep = piece
                    continue
                left_shoulder_x = piece['left_shoulder_x']
                left_shoulder_y = piece['left_shoulder_y']
                old_left_shoulder_x = last_timestep['left_shoulder_x']
                old_left_shoulder_y = last_timestep['left_shoulder_y']
                ls_dist = pt_dist([left_shoulder_x, left_shoulder_y], [old_left_shoulder_x, old_left_shoulder_y])

                right_shoulder_x = piece['right_shoulder_x']
                right_shoulder_y = piece['right_shoulder_y']
                old_right_shoulder_x = last_timestep['right_shoulder_x']
                old_right_shoulder_y = last_timestep['right_shoulder_y']
                rs_dist = pt_dist([right_shoulder_x, right_shoulder_y], [old_right_shoulder_x, old_right_shoulder_y])

                left_wrist_x = piece['left_wrist_x']
                left_wrist_y = piece['left_wrist_y']
                old_left_wrist_x = last_timestep['left_wrist_x']
                old_left_wrist_y = last_timestep['left_wrist_y']
                lw_dist = pt_dist([left_wrist_x, left_wrist_y], [old_left_wrist_x, old_left_wrist_y])

                right_wrist_x = piece['right_wrist_x']
                right_wrist_y = piece['right_wrist_y']
                old_right_wrist_x = last_timestep['right_wrist_x']
                old_right_wrist_y = last_timestep['right_wrist_y']
                rw_dist = pt_dist([right_wrist_x, right_wrist_y], [old_right_wrist_x, old_right_wrist_y])

                left_elbow_x = piece['left_elbow_x']
                left_elbow_y = piece['left_elbow_y']
                old_left_elbow_x = last_timestep['left_elbow_x']
                old_left_elbow_y = last_timestep['left_elbow_y']
                le_dist = pt_dist([left_elbow_x, left_elbow_y], [old_left_elbow_x, old_left_elbow_y])

                right_elbow_x = piece['right_elbow_x']
                right_elbow_y = piece['right_elbow_y']
                old_right_elbow_x = last_timestep['right_elbow_x']
                old_right_elbow_y = last_timestep['right_elbow_y']
                re_dist = pt_dist([right_elbow_x, right_elbow_y], [old_right_elbow_x, old_right_elbow_y])

                left_eye_x = piece['left_eye_x']
                left_eye_y = piece['left_eye_y']
                old_left_eye_x = last_timestep['left_eye_x']
                old_left_eye_y = last_timestep['left_eye_y']
                ley_dist = pt_dist([left_eye_x, left_eye_y], [old_left_eye_x, old_left_eye_y])

                right_eye_x = piece['right_eye_x']
                right_eye_y = piece['right_eye_y']
                old_right_eye_x = last_timestep['right_eye_x']
                old_right_eye_y = last_timestep['right_eye_y']
                rey_dist = pt_dist([right_eye_x, right_eye_y], [old_right_eye_x, old_right_eye_y])

                nose_x = piece['nose_x']
                nose_y = piece['nose_y']
                old_nose_x = last_timestep['nose_x']
                old_nose_y = last_timestep['nose_y']
                no_dist = pt_dist([nose_x, nose_y], [old_nose_x, old_nose_y])

                neck_x = piece['neck_x']
                neck_y = piece['neck_y']
                old_neck_x = last_timestep['neck_x']
                old_neck_y = last_timestep['neck_y']
                ne_dist = pt_dist([neck_x, neck_y], [old_neck_x, old_neck_y])
                blank_image = np.ones((1080, 1920, 3), np.uint8) * 255

                if right_shoulder_x < 1.7e300 and right_shoulder_y < 1.7e300:
                    cv2.circle(blank_image, (int(right_shoulder_y), int(right_shoulder_x)), 5, (0, 0, 255), thickness=4)
                if left_shoulder_x < 1.7e300 and left_shoulder_y < 1.7e300:
                    cv2.circle(blank_image, (int(left_shoulder_y), int(left_shoulder_x)), 5, (0, 255, 0), thickness=4)
                if left_wrist_y < 1.7e300 and left_wrist_x < 1.7e300:
                    cv2.circle(blank_image, (int(left_wrist_y), int(left_wrist_x)), 5, (0, 255, 0), thickness=4)
                if right_wrist_x < 1.7e300 and right_wrist_y < 1.7e300:
                    cv2.circle(blank_image, (int(right_wrist_y), int(right_wrist_x)), 5, (0, 0, 255), thickness=4)
                if left_elbow_x < 1.7e300 and left_elbow_y < 1.7e300:
                    cv2.circle(blank_image, (int(left_elbow_y), int(left_elbow_x)), 5, (0, 255, 0), thickness=4)
                if right_elbow_x < 1.7e300 and right_elbow_y < 1.7e300:
                    cv2.circle(blank_image, (int(right_elbow_y), int(right_elbow_x)), 5, (0, 0, 255), thickness=4)
                if right_eye_x < 1.7e300 and right_eye_y < 1.7e300:
                    cv2.circle(blank_image, (int(right_eye_y), int(right_eye_x)), 5, (0, 255, 255), thickness=4)
                if left_eye_x < 1.7e300 and left_eye_y < 1.7e300:
                    cv2.circle(blank_image, (int(left_eye_y), int(left_eye_x)), 5, (255, 255, 0), thickness=4)
                if nose_x < 1.7e300 and nose_y < 1.7e300:
                    cv2.circle(blank_image, (int(nose_y), int(nose_x)), 5, (255, 0, 0), thickness=4)
                if neck_x < 1.7e300 and neck_y < 1.7e300:
                    cv2.circle(blank_image, (int(neck_y), int(neck_x)), 5, (255, 0, 255), thickness=4)

                nose_vec_x = piece['nose_x'] - piece['neck_x']
                nose_vec_y = piece['nose_y'] - piece['neck_y']
                right_eye_vec_x = piece['right_eye_x'] - piece['neck_x']
                right_eye_vec_y = piece['right_eye_y'] - piece['neck_y']
                left_eye_vec_x = piece['left_eye_x'] - piece['neck_x']
                left_eye_vec_y = piece['left_eye_y'] - piece['neck_y']
                right_eye_nose_angle = angle_between([right_eye_vec_x, right_eye_vec_y], [nose_vec_x, nose_vec_y])
                left_eye_nose_angle = angle_between([left_eye_vec_x, left_eye_vec_y], [nose_vec_x, nose_vec_y])
                print 'Right Nose Angle:', right_eye_nose_angle
                print 'Left Nose Angle:', left_eye_nose_angle

                left_elbow_vec_x = piece['left_elbow_x'] - piece['left_shoulder_x']
                left_elbow_vec_y = piece['left_elbow_y'] - piece['left_shoulder_y']
                left_to_right_shoulder_x = piece['right_shoulder_x'] - piece['left_shoulder_x']
                left_to_right_shoulder_y = piece['right_shoulder_y'] - piece['left_shoulder_y']
                left_shoulder_angle = angle_between([left_elbow_vec_x, left_elbow_vec_y],
                                                    [left_to_right_shoulder_x, left_to_right_shoulder_y])
                print 'Left Shoulder Angle:', left_shoulder_angle
                left_wrist_vec_x = piece['left_wrist_x'] - piece['left_elbow_x']
                left_wrist_vec_y = piece['left_wrist_y'] - piece['left_elbow_y']
                left_elbow_angle = angle_between([left_wrist_vec_x, left_wrist_vec_y],
                                                 [-1 * left_elbow_vec_x, -1 * left_elbow_vec_y])
                print 'Left Elbow Angle:', left_elbow_angle
                # Want distance for shoulder -> wrist. also want angle
                left_shoulder_wrist_x = piece['left_wrist_x'] - piece['left_shoulder_x']
                left_shoulder_wrist_y = piece['left_wrist_y'] - piece['left_shoulder_y']
                left_wrist_shoulder_angle = angle_between([left_shoulder_wrist_x, left_shoulder_wrist_y],
                                                          [left_to_right_shoulder_x, left_to_right_shoulder_y])
                print 'Left Shoulder-Wrist Angle:', left_wrist_shoulder_angle

                right_to_left_shoulder_vec_x = piece['left_shoulder_x'] - piece['right_shoulder_x']
                right_to_left_shoulder_vec_y = piece['left_shoulder_y'] - piece['right_shoulder_y']
                right_elbow_vec_x = piece['right_elbow_x'] - piece['right_shoulder_x']
                right_elbow_vec_y = piece['right_elbow_y'] - piece['right_shoulder_y']
                right_shoulder_angle = angle_between([right_elbow_vec_x, right_elbow_vec_y],
                                                     [right_to_left_shoulder_vec_x, right_to_left_shoulder_vec_y])
                print 'Right Shoulder Angle:', right_shoulder_angle

                right_wrist_vec_x = piece['right_wrist_x'] - piece['right_elbow_x']
                right_wrist_vec_y = piece['right_wrist_y'] - piece['right_elbow_y']
                right_elbow_angle = angle_between([right_wrist_vec_x, right_wrist_vec_y],
                                                  [-1 * right_elbow_vec_x, -1 * right_elbow_vec_y])
                print 'Right Elbow Angle', right_elbow_angle

                # Want distance & angle for shoulder -> wrist
                right_shoulder_wrist_x = piece['right_wrist_x'] - piece['right_shoulder_x']
                right_shoulder_wrist_y = piece['right_wrist_y'] - piece['right_shoulder_y']
                right_wrist_shoulder_angle = angle_between([right_shoulder_wrist_x, right_shoulder_wrist_y],
                                                           [right_to_left_shoulder_vec_x, right_to_left_shoulder_vec_y])
                print 'Right Shoulder-Wrist Angle:', right_wrist_shoulder_angle
                # cv2.circle(blank_image, (int(right_shoulder_y), int(right_shoulder_x)), 5, (0, 0, 255), thickness=4)
                # cv2.circle(blank_image, (int(left_shoulder_y), int(left_shoulder_x)), 5, (0, 255, 0), thickness=4)
                # cv2.circle(blank_image, (int(left_wrist_y), int(left_wrist_x)), 5, (0, 255, 0), thickness=4)
                # cv2.circle(blank_image, (int(right_wrist_y), int(right_wrist_x)), 5, (0, 0, 255), thickness=4)
                # cv2.circle(blank_image, (int(left_elbow_y), int(left_elbow_x)), 5, (0, 255, 0), thickness=4)
                # cv2.circle(blank_image, (int(right_elbow_y), int(right_elbow_x)), 5, (0, 0, 255), thickness=4)
                # cv2.circle(blank_image, (int(right_eye_y), int(right_eye_x)), 5, (0, 0, 255), thickness=4)
                # cv2.circle(blank_image, (int(left_eye_y), int(left_eye_x)), 5, (0, 255, 0), thickness=4)
                # cv2.circle(blank_image, (int(nose_y), int(nose_x)), 5, (255, 0, 0), thickness=4)
                # cv2.circle(blank_image, (int(neck_y), int(neck_x)), 5, (255, 0, 255), thickness=4)
                cv2.imshow('skele', blank_image)
                cv2.waitKey(0)

                if ls_dist < 1.7e300:
                    avg_l_s += ls_dist
                    lsc += 1
                if rs_dist < 1.7e300:
                    avg_r_s += rs_dist
                    rsc += 1
                if re_dist < 1.7e300:
                    avg_r_e += re_dist
                    rec += 1
                if le_dist < 1.7e300:
                    avg_l_e += le_dist
                    lec += 1
                if rw_dist < 1.7e300:
                    avg_r_w += rw_dist
                    rwc += 1
                if lw_dist < 1.7e300:
                    avg_l_w += lw_dist
                    lwc += 1
                if ley_dist < 1.7e300:
                    avg_l_ey += ley_dist
                    leyc += 1
                if rey_dist < 1.7e300:
                    avg_r_ey += rey_dist
                    reyc += 1
                if no_dist < 1.7e300:
                    avg_no += no_dist
                    noc += 1
                if ne_dist < 1.7e300:
                    avg_ne += ne_dist
                    nec += 1
    print 'left shoulder avg:', avg_l_s/lsc
    print 'right shoulder avg:', avg_r_s/rsc
    print 'left elbow avg:', avg_l_e/lec
    print 'right elbow avg:', avg_r_e/rec
    print 'left wrist avg:', avg_l_w/lwc
    print 'right wrist avg:', avg_r_w/rwc
    print 'left eye avg:', avg_l_ey/leyc
    print 'right eye avg:', avg_r_ey/reyc
    print 'nose avg:', avg_no/noc
    print 'neck avg:', avg_ne/nec


def plot_angles():
    annots_dir = '/Users/andrewsilva/my_annots_trimmed'
    one_angles = []
    two_angles = []
    three_angles = []
    four_angles = []
    angle_vals = []
    label_vals = []
    color_vals = []
    colors = [(0, 1.0, 1.0), (1.0, 0, 0), (1.0, 1.0, 0), (0, 1.0, 0), (0, 0, 1.0)]
    for filename in os.listdir(annots_dir):
        af = open(os.path.join(annots_dir, filename), 'rb')
        annotation = pickle.load(af)
        # For each person in the segmentation
        for person in annotation:
            sorted_ann = sorted(annotation[person], key=lambda x: x['timestamp'])
            for piece in sorted_ann:
                # add_angle = piece['right_eye_angle']
                nose = [piece['nose_x'] - piece['neck_x'], piece['nose_y'] - piece['neck_y']]
                shoulder = [piece['right_shoulder_x'] - piece['neck_x'], piece['right_shoulder_y']-piece['neck_y']]
                add_angle = angle_between(nose, shoulder)
                if add_angle >= 1.7e300:
                    continue
                if piece['value'] == 1:
                    one_angles.append(add_angle)
                elif piece['value'] == 2:
                    two_angles.append(add_angle)
                elif piece['value'] == 3:
                    three_angles.append(add_angle)
                elif piece['value'] == 4:
                    four_angles.append(add_angle)
                angle_vals.append(add_angle)
                label_vals.append(piece['value'])
                color_vals.append(colors[piece['value']])
    plt.scatter(label_vals, angle_vals, c=color_vals)
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def train_rel_cpm(model_name, use_prior, clf, dropout_val = 0.75):
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
            # one_back = []
            # two_back = []
            # three_back = []
            # for i in range(29):
            #     one_back.append(-1)
            #     two_back.append(-1)
            #     three_back.append(-1)

            for piece in sorted_ann:
                new_input_data = []
                for key, value in piece.iteritems():
                    if key in bad_keys:
                        continue
                    if value >= 1.7e300:
                        piece[key] = -1
                label = piece['value']
                if label == 0:
                    last_value = -1
                    continue
                new_input_data.append(piece['right_wrist_angle'])       # 0
                new_input_data.append(piece['right_elbow_angle'])       # 1
                new_input_data.append(piece['left_wrist_angle'])        # 2
                new_input_data.append(piece['left_elbow_angle'])        # 3
                new_input_data.append(piece['shoulder_vec_x'])          # 4
                new_input_data.append(piece['shoulder_vec_y'])          # 5
                new_input_data.append(piece['left_eye_angle'])          # 6
                new_input_data.append(piece['right_eye_angle'])         # 7
                new_input_data.append(piece['eye_vec_x'])               # 8
                new_input_data.append(piece['eye_vec_y'])               # 9
                new_input_data.append(piece['right_shoulder_angle'])    # 10
                new_input_data.append(piece['left_shoulder_angle'])     # 11
                new_input_data.append(piece['nose_vec_y'])              # 12
                new_input_data.append(piece['nose_vec_x'])              # 13
                new_input_data.append(piece['left_arm_vec_x'])          # 14
                new_input_data.append(piece['left_arm_vec_y'])          # 15
                new_input_data.append(piece['right_arm_vec_x'])         # 16
                new_input_data.append(piece['right_arm_vec_y'])         # 17
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
                for item in piece['objects']:
                    if item == 'book':
                        new_input_data[19] += 1
                    elif item == 'bottle':
                        new_input_data[20] += 1
                    elif item == 'bowl':
                        new_input_data[21] += 1
                    elif item == 'cup':
                        new_input_data[22] += 1
                    elif item == 'laptop':
                        new_input_data[23] += 1
                    elif item == 'cell phone':
                        new_input_data[24] += 1
                    elif item == 'blocks':
                        new_input_data[25] += 1
                    elif item == 'tablet':
                        new_input_data[26] += 1
                    else:
                        new_input_data[27] += 1
                if use_prior:
                    if np.random.rand() > dropout_val:
                        new_input_data.append(last_value)
                    else:
                        new_input_data.append(-1)

                # holder = new_input_data[:]
                # for element in one_back:
                #     new_input_data.append(element)
                # for element in two_back:
                #     new_input_data.append(element)
                # for element in three_back:
                #     new_input_data.append(element)
                # three_back = two_back[:]
                # two_back = one_back[:]
                # one_back = holder[:]

                last_value = label
                X.append(new_input_data)
                Y.append(label)
    X = np.array(X)
    Y = np.array(Y)
    X_train, X_test, y_train, y_test = non_shuffling_train_test_split(X, Y, test_size=0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf.fit(X_train, y_train)
    joblib.dump(clf, model_name)
    # VISUALIZE FEATURE IMPORTANCE
    # importances = clf.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in clf.estimators_],
    #              axis=0)
    # indices = np.argsort(importances)[::-1]
    #
    # # Print the feature ranking
    # print("Feature ranking:")
    # X = np.array(X)
    # for f in range(X.shape[1]):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    #
    # # Plot the feature importances of the forest
    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar(range(X.shape[1]), importances[indices],
    #         color="r", yerr=std[indices], align="center")
    # plt.xticks(range(X.shape[1]), indices)
    # plt.xlim([-1, X.shape[1]])
    # plt.show()
    # END FEATURE IMPORTANCE VISUALIZATION
    # TODO manual k-fold stuff, predict single file and propagate predictions.
    last_prediction = -1
    y_pred = []
    for sample in X_test:
        # predict, replace element[28] with prediction.
        if use_prior:
            sample[-1] = last_prediction
        sample = np.reshape(sample, (1, -1))
        prediction = clf.predict(sample)
        last_prediction = prediction
        y_pred.append(prediction)

    # y_pred2 = clf.predict(X_test)
    # print y_pred
    # print y_pred2
    class_names = ['1', '2', '3', '4']
    # my_matrix = confusion_matrix(y_test, y_pred)
    # print my_matrix
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names,
    #                       title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
    return scaler


def test_temporal_model_rel(scaler, model_name, use_prior):
    annots_dir = '/Users/andrewsilva/my_rel_data/'
    true_y = []
    pred_y = []
    bad_keys = ['bb_y', 'bb_x', 'gaze_timestamp', 'pos_frame', 'send_timestamp', 'bb_height', 'pos_timestamp',
                'timestamp', 'bb_width', 'objects_timestamp', 'pos_z', 'pos_x', 'pos_y', 'name', 'pose_timestamp',
                'objects']
    clf = joblib.load(model_name)
    for filename in os.listdir(annots_dir):
        af = open(os.path.join(annots_dir, filename), 'rb')
        annotation = pickle.load(af)
        # For each person in the segmentation
        for person in annotation:
            last_value = -1
            sorted_ann = sorted(annotation[person], key=lambda x: x['timestamp'])

            # previous_timestep = []
            # for i in range(28):
            #     previous_timestep.append(-1)

            for piece in sorted_ann:
                for key, value in piece.iteritems():
                    if key in bad_keys:
                        continue
                    if value >= 1.7e300:
                        piece[key] = -1
                true_label = piece['value']
                if true_label == 0:
                    last_value = -1
                    continue
                new_input_data = []
                new_input_data.append(piece['right_wrist_angle'])
                new_input_data.append(piece['right_elbow_angle'])
                new_input_data.append(piece['left_wrist_angle'])
                new_input_data.append(piece['left_elbow_angle'])
                new_input_data.append(piece['shoulder_vec_x'])
                new_input_data.append(piece['shoulder_vec_y'])
                new_input_data.append(piece['left_eye_angle'])
                new_input_data.append(piece['right_eye_angle'])
                new_input_data.append(piece['eye_vec_x'])
                new_input_data.append(piece['eye_vec_y'])
                new_input_data.append(piece['right_shoulder_angle'])
                new_input_data.append(piece['left_shoulder_angle'])
                new_input_data.append(piece['nose_vec_y'])
                new_input_data.append(piece['nose_vec_x'])
                new_input_data.append(piece['left_arm_vec_x'])
                new_input_data.append(piece['left_arm_vec_y'])
                new_input_data.append(piece['right_arm_vec_x'])
                new_input_data.append(piece['right_arm_vec_y'])
                new_input_data.append(piece['gaze'])
                new_input_data.append(0) # book 19
                new_input_data.append(0) # bottle 20
                new_input_data.append(0) # bowl 21
                new_input_data.append(0) # cup 22
                new_input_data.append(0) # laptop 23
                new_input_data.append(0) # cell phone 24
                new_input_data.append(0) # blocks 25
                new_input_data.append(0) # tablet 26
                new_input_data.append(0) # unknown 27
                for item in piece['objects']:
                    if item == 'book':
                        new_input_data[19] += 1
                    elif item == 'bottle':
                        new_input_data[20] += 1
                    elif item == 'bowl':
                        new_input_data[21] += 1
                    elif item == 'cup':
                        new_input_data[22] += 1
                    elif item == 'laptop':
                        new_input_data[23] += 1
                    elif item == 'cell phone':
                        new_input_data[24] += 1
                    elif item == 'blocks':
                        new_input_data[25] += 1
                    elif item == 'tablet':
                        new_input_data[26] += 1
                    else:
                        new_input_data[27] += 1
                if use_prior:
                    new_input_data.append(last_value)

                # holder = new_input_data[:]
                # for element in previous_timestep:
                #     new_input_data.append(element)
                # previous_timestep = holder[:]

                new_input_data = np.reshape(new_input_data, (1, -1))
                new_input_data = scaler.transform(new_input_data)
                true_y.append(true_label)
                prediction = clf.predict(new_input_data)
                last_value = prediction
                pred_y.append(prediction)
    class_names = ['1', '2', '3', '4']
    # my_matrix = confusion_matrix(y_test, y_pred)
    # print my_matrix
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(true_y, pred_y)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()


def train_raw_cpm(model_name, use_prior, clf, dropout_val = 0.75):
    annots_dir = '/Users/andrewsilva/my_annots_trimmed'
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
            sorted_ann = sorted(annotation[person], key=lambda x: x['timestamp'])

            # one_back = []
            # two_back = []
            # three_back = []
            # if use_prior:
            #     input_len = 31
            # else:
            #     input_len = 30
            # for i in range(input_len):
            #     one_back.append(-1)
                # two_back.append(-1)
                # three_back.append(-1)

            for piece in sorted_ann:
                new_input_data = []
                for key, value in piece.iteritems():
                    if key in bad_keys:
                        continue
                    if value >= 1.7e300:
                        piece[key] = -1
                label = piece['value']
                if label == 0:
                    last_value = -1
                    continue
                new_input_data.append(piece['left_shoulder_x'])
                new_input_data.append(piece['left_shoulder_y'])
                new_input_data.append(piece['left_elbow_x'])
                new_input_data.append(piece['left_elbow_y'])
                new_input_data.append(piece['left_wrist_x'])
                new_input_data.append(piece['left_wrist_y'])
                new_input_data.append(piece['left_eye_x'])
                new_input_data.append(piece['left_eye_y'])
                new_input_data.append(piece['right_shoulder_x'])
                new_input_data.append(piece['right_shoulder_y'])
                new_input_data.append(piece['right_elbow_x'])
                new_input_data.append(piece['right_elbow_y'])
                new_input_data.append(piece['right_wrist_x'])
                new_input_data.append(piece['right_wrist_y'])
                new_input_data.append(piece['right_eye_x'])
                new_input_data.append(piece['right_eye_y'])
                new_input_data.append(piece['nose_x'])
                new_input_data.append(piece['nose_y'])
                new_input_data.append(piece['neck_x'])
                new_input_data.append(piece['neck_y'])
                new_input_data.append(piece['gaze'])
                new_input_data.append(0)  # book 21
                new_input_data.append(0)  # bottle 22
                new_input_data.append(0)  # bowl 23
                new_input_data.append(0)  # cup 24
                new_input_data.append(0)  # laptop 25
                new_input_data.append(0)  # cell phone 26
                new_input_data.append(0)  # blocks 27
                new_input_data.append(0)  # tablet 28
                new_input_data.append(0)  # unknown 29
                for item in piece['objects']:
                    if item == 'book':
                        new_input_data[21] += 1
                    elif item == 'bottle':
                        new_input_data[22] += 1
                    elif item == 'bowl':
                        new_input_data[23] += 1
                    elif item == 'cup':
                        new_input_data[24] += 1
                    elif item == 'laptop':
                        new_input_data[25] += 1
                    elif item == 'cell phone':
                        new_input_data[26] += 1
                    elif item == 'blocks':
                        new_input_data[27] += 1
                    elif item == 'tablet':
                        new_input_data[28] += 1
                    else:
                        new_input_data[29] += 1
                if use_prior:
                    if np.random.rand() > dropout_val:
                        new_input_data.append(last_value)
                    else:
                        new_input_data.append(-1)
                last_value = label

                # holder = new_input_data[:]
                # for element in one_back:
                #     new_input_data.append(element)
                # for element in two_back:
                #     new_input_data.append(element)
                # for element in three_back:
                #     new_input_data.append(element)
                # three_back = two_back[:]
                # two_back = one_back[:]
                # one_back = holder[:]

                X.append(new_input_data)
                Y.append(label)
    X = np.array(X)
    Y = np.array(Y)
    X_train, X_test, y_train, y_test = non_shuffling_train_test_split(X, Y, test_size=0.33)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf.fit(X_train, y_train)
    joblib.dump(clf, model_name)
    # clf.fit(X_train, y_train)
    last_prediction = -1
    # two_preds = -1
    # three_preds = -1
    # y_pred = []
    # for sample in X_test:
    #     # predict, replace element[28] with prediction.
    #     if use_prior:
    #         sample[30] = last_prediction
    #         # sample[61] = two_preds
    #         # sample[92] = three_preds
    #         # three_preds = two_preds
    #         # two_preds = last_prediction
    #     sample = np.reshape(sample, (1, -1))
    #     sample = scaler.transform(sample)
    #     prediction = clf.predict(sample)
    #     last_prediction = prediction
    #     y_pred.append(prediction)

    y_pred = clf.predict(X_test)
    class_names = ['1', '2', '3', '4']
    # my_matrix = confusion_matrix(y_test, y_pred)
    # print my_matrix
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names,
    #                       title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
    return scaler


def test_temporal_model_raw(scaler, model_name, use_prior):
    annots_dir = '/Users/andrewsilva/my_annots_trimmed/'
    true_y = []
    pred_y = []
    bad_keys = ['bb_y', 'bb_x', 'gaze_timestamp', 'pos_frame', 'send_timestamp', 'bb_height', 'pos_timestamp',
                'timestamp', 'bb_width', 'objects_timestamp', 'pos_z', 'pos_x', 'pos_y', 'name', 'pose_timestamp',
                'objects']
    clf = joblib.load(model_name)
    for filename in os.listdir(annots_dir):
        af = open(os.path.join(annots_dir, filename), 'rb')
        annotation = pickle.load(af)
        # For each person in the segmentation
        for person in annotation:
            last_value = -1
            sorted_ann = sorted(annotation[person], key=lambda x: x['timestamp'])
            for piece in sorted_ann:
                for key, value in piece.iteritems():
                    if key in bad_keys:
                        continue
                    if value >= 1.7e300:
                        piece[key] = -1
                true_label = piece['value']
                if true_label == 0:
                    last_value = -1
                    continue
                new_input_data = []
                new_input_data.append(piece['left_shoulder_x'])
                new_input_data.append(piece['left_shoulder_y'])
                new_input_data.append(piece['left_elbow_x'])
                new_input_data.append(piece['left_elbow_y'])
                new_input_data.append(piece['left_wrist_x'])
                new_input_data.append(piece['left_wrist_y'])
                new_input_data.append(piece['left_eye_x'])
                new_input_data.append(piece['left_eye_y'])
                new_input_data.append(piece['right_shoulder_x'])
                new_input_data.append(piece['right_shoulder_y'])
                new_input_data.append(piece['right_elbow_x'])
                new_input_data.append(piece['right_elbow_y'])
                new_input_data.append(piece['right_wrist_x'])
                new_input_data.append(piece['right_wrist_y'])
                new_input_data.append(piece['right_eye_x'])
                new_input_data.append(piece['right_eye_y'])
                new_input_data.append(piece['nose_x'])
                new_input_data.append(piece['nose_y'])
                new_input_data.append(piece['neck_x'])
                new_input_data.append(piece['neck_y'])
                new_input_data.append(piece['gaze'])
                new_input_data.append(0)  # book 21
                new_input_data.append(0)  # bottle 22
                new_input_data.append(0)  # bowl 23
                new_input_data.append(0)  # cup 24
                new_input_data.append(0)  # laptop 25
                new_input_data.append(0)  # cell phone 26
                new_input_data.append(0)  # blocks 27
                new_input_data.append(0)  # tablet 28
                new_input_data.append(0)  # unknown 29
                for item in piece['objects']:
                    if item == 'book':
                        new_input_data[21] += 1
                    elif item == 'bottle':
                        new_input_data[22] += 1
                    elif item == 'bowl':
                        new_input_data[23] += 1
                    elif item == 'cup':
                        new_input_data[24] += 1
                    elif item == 'laptop':
                        new_input_data[25] += 1
                    elif item == 'cell phone':
                        new_input_data[26] += 1
                    elif item == 'blocks':
                        new_input_data[27] += 1
                    elif item == 'tablet':
                        new_input_data[28] += 1
                    else:
                        new_input_data[29] += 1
                if use_prior:
                    new_input_data.append(last_value)
                new_input_data = np.array(new_input_data).reshape((1, -1))
                new_input_data = scaler.transform(new_input_data)
                true_y.append(true_label)
                prediction = clf.predict(new_input_data)
                last_value = prediction
                pred_y.append(prediction)
    class_names = ['1', '2', '3', '4']
    # my_matrix = confusion_matrix(y_test, y_pred)
    # print my_matrix
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(true_y, pred_y)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()


def train_no_cpm(model_name, use_prior, clf, dropout_val = 0.75):
    annots_dir = '/Users/andrewsilva/my_annots_trimmed'
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
            sorted_ann = sorted(annotation[person], key=lambda x: x['timestamp'])
            for piece in sorted_ann:
                new_input_data = []
                for key, value in piece.iteritems():
                    if key in bad_keys:
                        continue
                    if value >= 1.7e300:
                        piece[key] = -1
                label = piece['value']
                if label == 0:
                    last_value = -1
                    continue
                new_input_data.append(piece['gaze'])
                new_input_data.append(0)  # book 1
                new_input_data.append(0)  # bottle 2
                new_input_data.append(0)  # bowl 3
                new_input_data.append(0)  # cup 4
                new_input_data.append(0)  # laptop 5
                new_input_data.append(0)  # cell phone 6
                new_input_data.append(0)  # blocks 7
                new_input_data.append(0)  # tablet 8
                new_input_data.append(0)  # unknown 9
                for item in piece['objects']:
                    if item == 'book':
                        new_input_data[1] += 1
                    elif item == 'bottle':
                        new_input_data[2] += 1
                    elif item == 'bowl':
                        new_input_data[3] += 1
                    elif item == 'cup':
                        new_input_data[4] += 1
                    elif item == 'laptop':
                        new_input_data[5] += 1
                    elif item == 'cell phone':
                        new_input_data[6] += 1
                    elif item == 'blocks':
                        new_input_data[7] += 1
                    elif item == 'tablet':
                        new_input_data[8] += 1
                    else:
                        new_input_data[9] += 1
                if use_prior:
                    if np.random.rand() > dropout_val:
                        new_input_data.append(last_value)
                    else:
                        new_input_data.append(-1)
                last_value = label
                X.append(new_input_data)
                Y.append(label)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf.fit(X_train, y_train)
    joblib.dump(clf, model_name)
    # clf.fit(X_train, y_train)

    last_prediction = -1
    y_pred = []
    for sample in X_test:
        # predict, replace element[28] with prediction.
        if use_prior:
            sample[-1] = last_prediction
        sample = np.reshape(sample, (1, -1))
        prediction = clf.predict(sample)
        last_prediction = prediction
        y_pred.append(prediction)

    # y_pred = clf.predict(X_test)
    class_names = ['1', '2', '3', '4']
    # my_matrix = confusion_matrix(y_test, y_pred)
    # print my_matrix
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names,
    #                       title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
    return scaler


def test_temporal_model_no_cpm(scaler, model_name, use_prior):
    annots_dir = '/Users/andrewsilva/my_annots_trimmed/'
    true_y = []
    pred_y = []
    bad_keys = ['bb_y', 'bb_x', 'gaze_timestamp', 'pos_frame', 'send_timestamp', 'bb_height', 'pos_timestamp',
                'timestamp', 'bb_width', 'objects_timestamp', 'pos_z', 'pos_x', 'pos_y', 'name', 'pose_timestamp',
                'objects']
    clf = joblib.load(model_name)
    for filename in os.listdir(annots_dir):
        af = open(os.path.join(annots_dir, filename), 'rb')
        annotation = pickle.load(af)
        # For each person in the segmentation
        for person in annotation:
            last_value = -1
            sorted_ann = sorted(annotation[person], key=lambda x: x['timestamp'])
            for piece in sorted_ann:
                for key, value in piece.iteritems():
                    if key in bad_keys:
                        continue
                    if value >= 1.7e300:
                        piece[key] = -1
                true_label = piece['value']
                if true_label == 0:
                    last_value = -1
                    continue
                new_input_data = []
                new_input_data.append(piece['gaze'])
                new_input_data.append(0)  # book 1
                new_input_data.append(0)  # bottle 2
                new_input_data.append(0)  # bowl 3
                new_input_data.append(0)  # cup 4
                new_input_data.append(0)  # laptop 5
                new_input_data.append(0)  # cell phone 6
                new_input_data.append(0)  # blocks 7
                new_input_data.append(0)  # tablet 8
                new_input_data.append(0)  # unknown 9
                for item in piece['objects']:
                    if item == 'book':
                        new_input_data[1] += 1
                    elif item == 'bottle':
                        new_input_data[2] += 1
                    elif item == 'bowl':
                        new_input_data[3] += 1
                    elif item == 'cup':
                        new_input_data[4] += 1
                    elif item == 'laptop':
                        new_input_data[5] += 1
                    elif item == 'cell phone':
                        new_input_data[6] += 1
                    elif item == 'blocks':
                        new_input_data[7] += 1
                    elif item == 'tablet':
                        new_input_data[8] += 1
                    else:
                        new_input_data[9] += 1
                if use_prior:
                    new_input_data.append(last_value)
                new_input_data = np.array(new_input_data).reshape((1, -1))
                new_input_data = scaler.transform(new_input_data)
                true_y.append(true_label)
                prediction = clf.predict(new_input_data)
                last_value = prediction
                pred_y.append(prediction)
    class_names = ['1', '2', '3', '4']
    # my_matrix = confusion_matrix(y_test, y_pred)
    # print my_matrix
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(true_y, pred_y)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()


def train_binary_rel_cpm(model_name, use_prior, clf, dropout_val = 0.75):
    annots_dir = '/home/asilva/Data/sid_annots'
    X = []
    Y = []
    bad_keys = ['bb_y', 'bb_x', 'gaze_timestamp', 'pos_frame', 'send_timestamp', 'bb_height', 'pos_timestamp',
                'timestamp', 'bb_width', 'objects_timestamp', 'pos_z', 'pos_x', 'pos_y', 'name', 'pose_timestamp',
                'objects']
    for filename in os.listdir(annots_dir):
        if os.path.isdir(os.path.join(annots_dir, filename)):
            continue
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
                label = piece.get('value', None)
                if label is None:
                    print piece
                    continue
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
                Y.append({'filename': filename, 'timestamp': piece['timestamp'],
                          'pos_y': piece['pos_y'], 'value': label, 'name':piece['name']})

    X = np.array(X)
    Y_labels = np.array([x['value'] for x in Y])
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    fold_count = 1
    # TODO uncomment to load and test models
    # clf = joblib.load('MLPrel.pkl')
    # scaler = joblib.load('scaler.pkl')

    #### BELOW IS FOR IN SAMPLE TIMELINE IMAGING ####
    # scaler = StandardScaler()
    # scaler.fit_transform(X)
    # clf.fit(X, Y_labels)
    master_prediction_list = {}
    # for index in range(len(X)):
    #     x_in = X[index]
    #     x_in = x_in.reshape(1, -1)
    #     y_in = Y[index]
    #     y_pred = clf.predict(x_in)
    #     fn = y_in['filename']
    #     if fn not in master_prediction_list:
    #         master_prediction_list[fn] = {}
    #     if y_in['name'] not in master_prediction_list[fn]:
    #         master_prediction_list[fn][y_in['name']] = []
    #     master_prediction_list[fn][y_in['name']].append({'timestamp': y_in['timestamp'],
    #                                                     'pos_y': y_in['pos_y'], 'value': y_pred[0]})
    #### END FOR IN SAMPLE TIMELINE IMAGING ####

    for train_idx, test_idx in kf.split(X, Y_labels):
        # X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
        # y_train, y_test = Y_labels[train_idx].copy(), Y[test_idx]
        X_train = X[train_idx].copy()
        y_train = Y_labels[train_idx].copy()
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)
        clf.fit(X_train, y_train)
        for index in test_idx:
            x_in = scaler.transform(X[index])
            x_in = x_in.reshape(1, -1)
            y_in = Y[index]
            y_pred = clf.predict(x_in)
            fn = y_in['filename']
            if fn not in master_prediction_list:
                master_prediction_list[fn] = {}
            if y_in['name'] not in master_prediction_list[fn]:
                master_prediction_list[fn][y_in['name']] = []
            master_prediction_list[fn][y_in['name']].append({'timestamp': y_in['timestamp'],
                                                            'pos_y': y_in['pos_y'], 'value': y_pred[0]})
        # y_pred = clf.predict(X_test)
        # print 'Fold:', fold_count
        # print 'F1:', f1_score(y_test, y_pred)
        # print 'Accuracy:', accuracy_score(y_test, y_pred)
        # cm = confusion_matrix(y_test, y_pred)
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print cm
        # fold_count += 1
    out_dir = '/home/asilva/Data/predicted_annots'
    for filename in master_prediction_list.keys():
        pickle.dump(master_prediction_list[filename], open(os.path.join(out_dir, filename), 'wb'))
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=59)
    # X_train, X_test, y_train, y_test = non_shuffling_train_test_split(X, Y, test_size=.2)

    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    #
    # clf.fit(X_train, y_train)
    # TODO uncomment to save models
    # joblib.dump(clf, model_name)
    # joblib.dump(scaler, 'scaler.pkl')
    # VISUALIZE FEATURE IMPORTANCE
    # importances = clf.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in clf.estimators_],
    #              axis=0)
    # indices = np.argsort(importances)[::-1]
    #
    # # Print the feature ranking
    # print("Feature ranking:")
    # X = np.array(X)
    # for f in range(X.shape[1]):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    #
    # # Plot the feature importances of the forest
    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar(range(X.shape[1]), importances[indices],
    #         color="r", yerr=std[indices], align="center")
    # plt.xticks(range(X.shape[1]), indices)
    # plt.xlim([-1, X.shape[1]])
    # plt.show()
    # END FEATURE IMPORTANCE VISUALIZATION
    # TODO manual k-fold stuff, predict single file and propagate predictions.
    last_prediction = -1
    # y_pred = []
    # for sample in X_test:
    #     # predict, replace element[28] with prediction.
    #     if use_prior:
    #         sample[-1] = last_prediction
    #     sample = np.reshape(sample, (1, -1))
    #     prediction = clf.predict(sample)
    #     last_prediction = prediction
    #     y_pred.append(prediction)

    # y_pred = clf.predict(X_test)
    # print y_pred
    # class_names = ['0', '1']
    # my_matrix = confusion_matrix(y_test, y_pred)
    # print my_matrix
    # cnf_matrix = confusion_matrix(y_test, y_pred)
    # np.set_printoptions(precision=2)
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix')
    #
    # plt.show()
    # print 'F1: ', f1_score(y_test, y_pred)

    return scaler


def train_binary_raw_cpm(model_name, use_prior, clf, dropout_val = 0.75):
    annots_dir = '/Users/andrewsilva/my_annots_trimmed'
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
            sorted_ann = sorted(annotation[person], key=lambda x: x['timestamp'])

            # one_back = []
            # two_back = []
            # three_back = []
            # if use_prior:
            #     input_len = 31
            # else:
            #     input_len = 30
            # for i in range(input_len):
            #     one_back.append(-1)
                # two_back.append(-1)
                # three_back.append(-1)

            for piece in sorted_ann:
                new_input_data = []
                for key, value in piece.iteritems():
                    if key in bad_keys:
                        continue
                    if value >= 1.7e300:
                        piece[key] = -1
                label = piece['value']
                if label == 0:
                    last_value = -1
                    continue
                new_input_data.append(piece['left_shoulder_x'])
                new_input_data.append(piece['left_shoulder_y'])
                new_input_data.append(piece['left_elbow_x'])
                new_input_data.append(piece['left_elbow_y'])
                new_input_data.append(piece['left_wrist_x'])
                new_input_data.append(piece['left_wrist_y'])
                new_input_data.append(piece['left_eye_x'])
                new_input_data.append(piece['left_eye_y'])
                new_input_data.append(piece['right_shoulder_x'])
                new_input_data.append(piece['right_shoulder_y'])
                new_input_data.append(piece['right_elbow_x'])
                new_input_data.append(piece['right_elbow_y'])
                new_input_data.append(piece['right_wrist_x'])
                new_input_data.append(piece['right_wrist_y'])
                new_input_data.append(piece['right_eye_x'])
                new_input_data.append(piece['right_eye_y'])
                new_input_data.append(piece['nose_x'])
                new_input_data.append(piece['nose_y'])
                new_input_data.append(piece['neck_x'])
                new_input_data.append(piece['neck_y'])
                new_input_data.append(piece['gaze'])
                new_input_data.append(0)  # book 21
                new_input_data.append(0)  # bottle 22
                new_input_data.append(0)  # bowl 23
                new_input_data.append(0)  # cup 24
                new_input_data.append(0)  # laptop 25
                new_input_data.append(0)  # cell phone 26
                new_input_data.append(0)  # blocks 27
                new_input_data.append(0)  # tablet 28
                new_input_data.append(0)  # unknown 29
                for item in piece['objects']:
                    if item == 'book':
                        new_input_data[21] += 1
                    elif item == 'bottle':
                        new_input_data[22] += 1
                    elif item == 'bowl':
                        new_input_data[23] += 1
                    elif item == 'cup':
                        new_input_data[24] += 1
                    elif item == 'laptop':
                        new_input_data[25] += 1
                    elif item == 'cell phone':
                        new_input_data[26] += 1
                    elif item == 'blocks':
                        new_input_data[27] += 1
                    elif item == 'tablet':
                        new_input_data[28] += 1
                    else:
                        new_input_data[29] += 1
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

                # holder = new_input_data[:]
                # for element in one_back:
                #     new_input_data.append(element)
                # for element in two_back:
                #     new_input_data.append(element)
                # for element in three_back:
                #     new_input_data.append(element)
                # three_back = two_back[:]
                # two_back = one_back[:]
                # one_back = holder[:]

                X.append(new_input_data)
                Y.append(label)
    X = np.array(X)
    Y = np.array(Y)
    # TODO: K Fold CV
    # kf = StratifiedKFold(n_splits=10, shuffle=False)
    # fold_count = 1
    # for train_idx, test_idx in kf.split(X, Y):
    #     X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
    #     y_train, y_test = Y[train_idx].copy(), Y[test_idx].copy()
    #     scaler = StandardScaler()
    #     scaler.fit(X_train)
    #     X_train = scaler.transform(X_train)
    #     X_test = scaler.transform(X_test)
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_test)
    #     print 'Fold:', fold_count
    #     print 'F1:', f1_score(y_test, y_pred)
    #     fold_count += 1

    X_train, X_test, y_train, y_test = non_shuffling_train_test_split(X, Y, test_size=0.2)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf.fit(X_train, y_train)
    joblib.dump(clf, model_name)

    y_pred = clf.predict(X_test)
    class_names = ['0', '1']

    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
    print 'F1: ', f1_score(y_test, y_pred)
    return scaler


def train_binary_no_cpm(model_name, use_prior, clf, dropout_val=0.75):
    annots_dir = '/Users/andrewsilva/my_annots_trimmed'
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
            sorted_ann = sorted(annotation[person], key=lambda x: x['timestamp'])
            for piece in sorted_ann:
                new_input_data = []
                for key, value in piece.iteritems():
                    if key in bad_keys:
                        continue
                    if value >= 1.7e300:
                        piece[key] = -1
                label = piece['value']
                if label == 0:
                    last_value = -1
                    continue
                new_input_data.append(piece['gaze'])
                new_input_data.append(0)  # book 1
                new_input_data.append(0)  # bottle 2
                new_input_data.append(0)  # bowl 3
                new_input_data.append(0)  # cup 4
                new_input_data.append(0)  # laptop 5
                new_input_data.append(0)  # cell phone 6
                new_input_data.append(0)  # blocks 7
                new_input_data.append(0)  # tablet 8
                new_input_data.append(0)  # unknown 9
                for item in piece['objects']:
                    if item == 'book':
                        new_input_data[1] += 1
                    elif item == 'bottle':
                        new_input_data[2] += 1
                    elif item == 'bowl':
                        new_input_data[3] += 1
                    elif item == 'cup':
                        new_input_data[4] += 1
                    elif item == 'laptop':
                        new_input_data[5] += 1
                    elif item == 'cell phone':
                        new_input_data[6] += 1
                    elif item == 'blocks':
                        new_input_data[7] += 1
                    elif item == 'tablet':
                        new_input_data[8] += 1
                    else:
                        new_input_data[9] += 1
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
    # kf = StratifiedKFold(n_splits=10, shuffle=False)
    # fold_count = 1
    # for train_idx, test_idx in kf.split(X, Y):
    #     X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
    #     y_train, y_test = Y[train_idx].copy(), Y[test_idx].copy()
    #     scaler = StandardScaler()
    #     scaler.fit(X_train)
    #     X_train = scaler.transform(X_train)
    #     X_test = scaler.transform(X_test)
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_test)
    #     print 'Fold:', fold_count
    #     print 'F1:', f1_score(y_test, y_pred)
    #     fold_count += 1
    X_train, X_test, y_train, y_test = non_shuffling_train_test_split(X, Y, test_size=0.25)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf.fit(X_train, y_train)
    joblib.dump(clf, model_name)
    # clf.fit(X_train, y_train)

    last_prediction = -1
    y_pred = []
    # for sample in X_test:
    #     # predict, replace element[28] with prediction.
    #     if use_prior:
    #         sample[-1] = last_prediction
    #     sample = np.reshape(sample, (1, -1))
    #     prediction = clf.predict(sample)
    #     last_prediction = prediction
    #     y_pred.append(prediction)

    y_pred = clf.predict(X_test)
    class_names = ['0', '1']
    # my_matrix = confusion_matrix(y_test, y_pred)
    # print my_matrix
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names,
    #                       title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
    print 'F1: ', f1_score(y_test, y_pred)
    return scaler


if __name__ == '__main__':
    if len(sys.argv) > 1:
        adir = sys.argv[1]
    model_name = 'MLPrel2.pkl'
    use_prior = False
    clf = RandomForestClassifier(class_weight="balanced_subsample", n_estimators=20)
    # clf = MLPClassifier(max_iter=40000, tol=1e-4)
    # clf = KNeighborsClassifier()
    # clf = svm.SVC(decision_function_shape='ovo')
    scaler = train_binary_rel_cpm(model_name, use_prior, clf, dropout_val=0.75)
    # test_temporal_model_rel(scaler, model_name, use_prior)
