# Silva validation of annotations

import sys
import re
import datetime
import time
import os
import pickle
import numpy as np
import cv2
from sklearn.metrics import f1_score

# Helpers
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_trial_pickles(trial_number, annotator):
    base_path = '/home/asilva/Data'
    if 'silva' == annotator.lower():
        trial_pwd = os.path.join(base_path, 'silva_trial_data', 'trial'+str(trial_number))
    elif 'sid' == annotator.lower():
        trial_pwd = os.path.join(base_path, 'separated', 'banerjs-29-08-17', 'trial'+str(trial_number))
    elif 'mlp' == annotator.lower():
        trial_pwd = os.path.join(base_path, 'separated', 'mlp_annots', 'trial'+str(trial_number))
    elif 'mlp_in_sample' == annotator.lower():
        trial_pwd = os.path.join(base_path, 'separated', 'mlp_in_sample', 'trial'+str(trial_number))
    elif 'rf_in_sample' == annotator.lower():
        trial_pwd = os.path.join(base_path, 'separated', 'rf_in_sample', 'trial'+str(trial_number))
    elif 'rf' == annotator.lower():
        trial_pwd = os.path.join(base_path, 'separated', 'rf_out_sample', 'trial'+str(trial_number))
    elif 'ldcrf' == annotator.lower():
        trial_pwd = os.path.join(base_path, 'ldcrf_out_sample', 'trial'+str(trial_number))
    elif 'auto' == annotator.lower():
        trial_pwd = os.path.join(base_path, 'separated', 'auto-29-08-17', 'trial'+str(trial_number))
    elif 'knn' == annotator.lower():
        trial_pwd = os.path.join(base_path, 'separated', 'knn_annots', 'trial'+str(trial_number))
    elif 'svm' == annotator.lower():
        trial_pwd = os.path.join(base_path, 'separated', 'svm_annots', 'trial'+str(trial_number))

    people = {'S': []}
    for annot_file in os.listdir(trial_pwd):
        af = open(os.path.join(trial_pwd, annot_file), 'rb')
        annotation = pickle.load(af)
        for person in annotation:
            timestamp_person_dict = {}
            sorted_ann = sorted(annotation[person], key = lambda x: x['timestamp'])
            for piece in sorted_ann:
                # print piece

                label = piece.get('value', 0)
                # For unknown interruptibility / missing people, just move on (if using model predictions, 0 is valid)
                if annotator.lower() == 'silva' or annotator.lower() == 'sid' or annotator.lower() == 'auto':
                    if label == 0:
                        continue

                # Binarize our labels (if using model predictions, assume already binarized)
                if annotator.lower() == 'silva' or annotator.lower() == 'sid':  # Manual annotations are [1, 2, 3, 4]
                    if label <= 2:
                        label = 0
                    else:
                        label = 1
                elif annotator.lower() == 'auto':  # Auto annotations are [-1, 1]
                    if label < 0:
                        label = 0
                timestamp_person_dict[piece['timestamp']] = label
            # Verified
            # if annotation[person][0]['pos_y'] <= 0:
            #     dict_name = 'R'
            # else:
            #     dict_name = 'L'
            people['S'].append(timestamp_person_dict)
    return people


def draw_int_timeline(image, int_list, starting_line=0, box_width=3, box_height=3, label='unknown', compare_to=None):
    # Assuming ints are 0 or 1
    # Image: image to draw on
    # starting_line: row of ints we're drawing on
    int_color = (255, 140, 0)   # Interruptible is blue
    busy_color = (0, 140, 255)  # Busy is orange
    x_ind = 0
    y_ind = starting_line*box_height
    cv2.putText(image, label, (x_ind, y_ind+box_height), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
    x_ind += 100
    if compare_to is not None:
        correct = 0
        preds = [x['value'] for x in int_list]
        gt = [x['value'] for x in compare_to]
        f1 = f1_score(gt, preds)
        accuracy = 100*float(correct)/float(len(int_list))
        cv2.putText(image, "F1=%.4f" % f1, (x_ind, y_ind+box_height), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
    x_ind += 100
    last_timestep = 0
    num_flips = 0
    last_val = int_list[0]['value']
    for index, int_val in enumerate(int_list):
        if int_val['timestamp'] - last_timestep > 30:
            col = (0, 0, 0)     # Start of a new session is black
            last_val = int_val['value']
            pt1 = (x_ind, y_ind)
            pt2 = (x_ind+box_width, y_ind+box_height)
            cv2.rectangle(image, pt1, pt2, color=col, thickness=-1)
            x_ind += box_width
        if int_val['value'] != last_val:
            num_flips += 1
        last_val = int_val['value']
        pt1 = (x_ind, y_ind)
        pt2 = (x_ind+box_width, y_ind+box_height)
        if int_val['value'] == 0:
            col = busy_color
        else:
            col = int_color

        last_timestep = int_val['timestamp']
        cv2.rectangle(image, pt1, pt2, color=col, thickness=-1)
        x_ind += box_width
    x_ind += 10
    num_flips = 100*float(num_flips)/float(len(int_list))
    cv2.putText(image, "%.2f" % num_flips + '%', (x_ind, y_ind+box_height), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))


def draw_ldcrf(image, int_list, gt_stamps, starting_line=0, box_width=3, box_height=3, label='unknown'):
    # Assuming ints are 0 or 1
    # Image: image to draw on
    # starting_line: row of ints we're drawing on
    int_color = (255, 140, 0)   # Interruptible is blue
    busy_color = (0, 140, 255)  # Busy is orange
    x_ind = 0
    y_ind = starting_line*box_height
    cv2.putText(image, label, (x_ind, y_ind+box_height), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
    x_ind += 200
    last_timestep = 0
    for int_val in gt_stamps:
        pt1 = (x_ind, y_ind)
        pt2 = (x_ind+box_width, y_ind+box_height)
        robot_entered = False
        if int_val['value'] == 0:
            col = busy_color
        else:
            col = int_color
        if int_val['timestamp'] - last_timestep > 60:
            col = (0, 0, 0)     # Start of a new session is black
            robot_entered = True
        last_timestep = int_val['timestamp']

        # Iterate over all the LDCRF list. If timestamp is not found, color in gray
        # If timestamp is found, color appropriately
        found = False
        for i in range(len(int_list)):
            if int_list[i]['timestamp'] == last_timestep:

                if int_list[i]['value'] == 0:
                    col = busy_color
                else:
                    col = int_color
                found = True
            if not found:
                col = (192, 192, 192)
        if robot_entered:
            col = (0, 0, 0)
        cv2.rectangle(image, pt1, pt2, color=col, thickness=-1)
        x_ind += box_width


def draw_timeline(annotator_name):
    for trial_id in range(1, 16):

        auto_annots = get_trial_pickles(trial_id, 'auto')
        annotations = get_trial_pickles(trial_id, annotator_name)
        mlp_annotations = get_trial_pickles(trial_id, 'mlp')
        mlp_in_sample = get_trial_pickles(trial_id, 'mlp_in_sample')
        rf_in_sample = get_trial_pickles(trial_id, 'rf_in_sample')
        svm_annots = get_trial_pickles(trial_id, 'svm')
        knn_annots = get_trial_pickles(trial_id, 'knn')
        rf_annotations = get_trial_pickles(trial_id, 'rf')
        # ldcrf_annotations = get_trial_pickles(trial_id, 'ldcrf')

        for participant in annotations.keys():
            part_ground = []
            part_annots = []
            part_mlp = []
            part_mlp_in_sample = []
            part_rf_in_sample = []
            part_rf = []
            part_svm = []
            part_knn = []
            # part_ldcrf = []

            for annot_array in auto_annots[participant]:
                for timestamp in annot_array.keys():
                    part_ground.append({'timestamp': timestamp, 'value': annot_array[timestamp]})

            for annot_array in annotations[participant]:
                for timestamp in annot_array.keys():
                    part_annots.append({'timestamp': timestamp, 'value': annot_array[timestamp]})

            for annot_array in mlp_annotations[participant]:
                for timestamp in annot_array.keys():
                    part_mlp.append({'timestamp': timestamp, 'value': annot_array[timestamp]})

            # for annot_array in mlp_in_sample[participant]:
            #     for timestamp in annot_array.keys():
            #         part_mlp_in_sample.append({'timestamp': timestamp, 'value': annot_array[timestamp]})
            #
            # for annot_array in rf_in_sample[participant]:
            #     for timestamp in annot_array.keys():
            #         part_rf_in_sample.append({'timestamp': timestamp, 'value': annot_array[timestamp]})

            for annot_array in rf_annotations[participant]:
                for timestamp in annot_array.keys():
                    part_rf.append({'timestamp': timestamp, 'value': annot_array[timestamp]})

            for annot_array in svm_annots[participant]:
                for timestamp in annot_array.keys():
                    part_svm.append({'timestamp': timestamp, 'value': annot_array[timestamp]})

            for annot_array in knn_annots[participant]:
                for timestamp in annot_array.keys():
                    part_knn.append({'timestamp': timestamp, 'value': annot_array[timestamp]})
            # for annot_array in ldcrf_annotations[participant]:
            #     for timestamp in annot_array.keys():
            #         part_ldcrf.append({'timestamp': timestamp, 'value': annot_array[timestamp]})

            part_ground = sorted(part_ground, key=lambda x: x['timestamp'])
            part_annots = sorted(part_annots, key=lambda x: x['timestamp'])
            part_mlp = sorted(part_mlp, key=lambda x: x['timestamp'])
            part_mlp_in_sample = sorted(part_mlp_in_sample, key=lambda x: x['timestamp'])
            part_rf_in_sample = sorted(part_rf_in_sample, key=lambda x: x['timestamp'])
            part_rf = sorted(part_rf, key=lambda x: x['timestamp'])
            # part_ldcrf = sorted(part_ldcrf, key=lambda x: x['timestamp'])
            print 'Trial: ', trial_id
            print 'Participant: ', participant
            box_height = 8
            box_width = 4
            img = np.ones((box_height*21, len(part_ground)*box_width + 300, 3), np.uint8)
            img *= 255
            draw_int_timeline(img, part_ground, starting_line=1, box_width=box_width,
                                   box_height=box_height, label='Ground Truth')
            draw_int_timeline(img, part_annots, starting_line=4, box_width=box_width,
                                   box_height=box_height, label='Human', compare_to=part_ground)
            draw_ldcrf(img, part_svm, part_ground, starting_line=7, box_width=box_width,
                                   box_height=box_height, label='SVM')
            draw_int_timeline(img, part_mlp, starting_line=10, box_width=box_width,
                                   box_height=box_height, label='MLP', compare_to=part_ground)
            draw_ldcrf(img, part_knn, part_ground, starting_line=13, box_width=box_width,
                                   box_height=box_height, label='KNN')
            draw_int_timeline(img, part_rf, starting_line=16, box_width=box_width,
                                   box_height=box_height, label='RF', compare_to=part_ground)
            # draw_ldcrf(img, part_ldcrf, part_ground, starting_line=19, box_width=box_width,
            #                        box_height=box_height, label='LDCRF')

            img_filename = 'trial_'+str(trial_id)+'_participant_'+participant+'.jpg'
            cv2.imwrite(img_filename, img)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        annotator_name = sys.argv[1]
    else:
        annotator_name = 'sid'
    draw_timeline(annotator_name)
