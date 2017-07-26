#!/usr/bin/env python
# This script takes in segmentations from /collections, pre-eyisting annotations from /my_pickle_out, and copies the
# values to corresponding segments in a new directory /copied_annotations
# Andrew Silva 05/30/17
import pickle
import os
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.externals import joblib


blank_image = np.ones((1080, 1920, 3), np.uint8) * 255


def non_shuffling_train_test_split(X, y, test_size=0.2):
    i = int((1 - test_size) * X.shape[0]) + 1
    X_train, X_test = np.split(X, [i])
    y_train, y_test = np.split(y, [i])
    return X_train, X_test, y_train, y_test


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


def train_binary_rel_cpm(model_name):
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

                if label <= 2:
                    label = 0
                else:
                    label = 1
                X.append(new_input_data)
                Y.append(label)

    X = np.array(X)
    Y = np.array(Y)
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    fold_count = 1
    # TODO uncomment to load and test models
    clf = joblib.load('MLPrel.pkl')
    scaler = joblib.load('scaler.pkl')
    for train_idx, test_idx in kf.split(X, Y):
        X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
        y_train, y_test = Y[train_idx].copy(), Y[test_idx].copy()
        X_test = scaler.transform(X_test)
        y_pred = clf.predict(X_test)
        print 'Fold:', fold_count
        print 'F1:', f1_score(y_test, y_pred)
        print 'Accuracy:', accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print cm
        fold_count += 1


if __name__ == '__main__':
    if len(sys.argv) > 1:
        adir = sys.argv[1]
    model_name = 'MLPrel.pkl'
    train_binary_rel_cpm(model_name)
