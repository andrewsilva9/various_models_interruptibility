import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle
import seaborn as sns
import pandas as pd


def import_rel_data(automated=False):
    if automated:
        annots_dir = '/home/asilva/Data/int_annotations/automated_annots'
    else:
        annots_dir = '/home/asilva/Data/int_annotations/my_rel_data'
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
                new_input_data.append(piece['left_eye_angle'])          # 4
                new_input_data.append(piece['right_eye_angle'])         # 5
                new_input_data.append(piece['right_shoulder_angle'])    # 6
                new_input_data.append(piece['left_shoulder_angle'])     # 7
                new_input_data.append(piece['nose_vec_y'])              # 8
                new_input_data.append(piece['nose_vec_x'])              # 9
                new_input_data.append(piece['gaze'])                    # 10
                new_input_data.append(0)                                # book 11
                new_input_data.append(0)                                # bottle 12
                new_input_data.append(0)                                # bowl 13
                new_input_data.append(0)                                # cup 14
                new_input_data.append(0)                                # laptop 15
                new_input_data.append(0)                                # cell phone 16
                new_input_data.append(0)                                # blocks 17
                new_input_data.append(0)                                # tablet 18
                # new_input_data.append(0)                                # unknown 19
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
                    elif item == 'block':
                        new_input_data[foi+6] += 1
                    elif item == 'tablet':
                        new_input_data[foi+7] += 1
                    else:
                        new_input_data[foi+8] += 1
                if not automated:
                    if label <= 2:
                        label = 0
                    else:
                        label = 1
                X.append(new_input_data)
                Y.append(label)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def get_stddvs(normalize=False, automated=True, label=2):
    X, Y = import_rel_data(automated=automated)
    # scaler = StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)
    np.set_printoptions(precision=6, suppress=True)
    if label < 2:
        X = X[Y==label]
    if normalize:
        maxes = np.max(X, axis=0)
        mins = np.min(X, axis=0)
        for i in range(X.shape[1]):
            if (maxes[i] - mins[i]) == 0:
                continue
            X[:, i] = (X[:, i] - mins[i]) / (maxes[i] - mins[i])
        # Drop unknown
        X = X[:, 0:-1]
    stddev = np.std(X, axis=0)
    mean = np.mean(X, axis=0)
    print 'Standard Deviation: ', stddev
    print 'Means: ', mean
    return stddev, mean


def vis_rel_data(scale=True, do_heatmap=True, do_variance=True, automated=True):
    X, Y = import_rel_data(automated=automated)
    if automated:
        labels = [-1, 1]
    else:
        labels = [0, 1]
    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
    n_comp = 19
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)
    colors = ['navy', 'darkorange']
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for color, i, target in zip(colors, labels, ['Uninterruptible', 'Interruptible']):
        ax.scatter(X_pca[Y==i, 0], X_pca[Y==i, 1], X_pca[Y==i, 2], color=color, lw=2, label=target)
    plt.title('PCA of binary interruptibility dataset')
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    if scale:
        plt.axis([-4, 4, -4, 6])
    else:
        plt.axis([-80, 150, -150, 150])
    if do_heatmap:
        plt.figure(figsize=(10, 6.5))
        plt.title('Feature Importances to PCA Features')
        sns.heatpltmap(np.log(pca.inverse_transform(np.eye(n_comp))), cmap="hot", cbar=False)
        plt.ylabel('principal component', fontsize=20)
        plt.xlabel('original feature index', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tick_params(axis='both', which='minor', labelsize=12)
    if do_variance:
        plt.figure(figsize=(10, 6.5))
        plt.semilogy(pca.explained_variance_ratio_, '--o')
        plt.xlabel('principal component', fontsize=20)
        plt.ylabel('explained variance', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tick_params(axis='both', which='minor', labelsize=12)
        plt.figure(figsize=(10, 6.5))
        plt.semilogy(np.square(X.std(axis=0)) / np.square(X.std(axis=0)).sum(), '--o', label='variance ratio')
        plt.semilogy(X.mean(axis=0) / np.square(X.mean(axis=0)).sum(), '--o', label='mean ratio')
        plt.xlabel('original feature', fontsize=20)
        plt.ylabel('variance', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tick_params(axis='both', which='minor', labelsize=12)
        plt.xlim([0, 20]);
        plt.legend(loc='lower left', fontsize=18)
    plt.show()


def vis_raw_data(scale=True):
    annots_dir = '/home/asilva/Data/int_annotations/my_annots_trimmed'
    X = []
    Y = []
    bad_keys = ['bb_y', 'bb_x', 'gaze_timestamp', 'pos_frame', 'send_timestamp', 'bb_height', 'pos_timestamp',
                'timestamp', 'bb_width', 'objects_timestamp', 'pos_z', 'pos_x', 'pos_y', 'name', 'pose_timestamp',
                'objects']
    scale = scale
    for filename in os.listdir(annots_dir):
        af = open(os.path.join(annots_dir, filename), 'rb')
        annotation = pickle.load(af)
        # For each person in the segmentation
        for person in annotation:
            sorted_ann = sorted(annotation[person], key=lambda x: x['timestamp'])
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
                    elif item == 'block':
                        new_input_data[27] += 1
                    elif item == 'tablet':
                        new_input_data[28] += 1
                    else:
                        new_input_data[29] += 1

                if label <= 2:
                    label = 0
                else:
                    label = 1

                X.append(new_input_data)
                Y.append(label)
    X = np.array(X)
    Y = np.array(Y)
    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
    n_comp = 30
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)
    colors = ['navy', 'darkorange']
    plt.figure(figsize=(8, 8))
    for color, i, target in zip(colors, [0, 1], ['Uninterruptible', 'Interruptible']):
        plt.scatter(X_pca[Y==i, 0], X_pca[Y==i, 1], color=color, lw=2, label=target)
    plt.title('PCA of binary interruptibility dataset')
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    if scale:
        plt.axis([-5, 5, -3.5, 4.5])
    else:
        plt.axis([-2500, 2400, -500, 1500])

    plt.figure(figsize=(15, 10))
    plt.title('Feature Importances to PCA Features')
    sns.heatmap(np.log(pca.inverse_transform(np.eye(n_comp))), cmap="hot", cbar=False)
    plt.ylabel('principal component', fontsize=20)
    plt.xlabel('original feature index', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.show()


def vis_automated_rel_data(scale=True):
    annots_dir = '/home/asilva/Data/int_annotations/automated_annots'
    X = []
    Y = []
    bad_keys = ['bb_y', 'bb_x', 'gaze_timestamp', 'pos_frame', 'send_timestamp', 'bb_height', 'pos_timestamp',
                'timestamp', 'bb_width', 'objects_timestamp', 'pos_z', 'pos_x', 'pos_y', 'name', 'pose_timestamp',
                'objects']
    scale = scale
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
                new_input_data.append(piece['left_eye_angle'])          # 4
                new_input_data.append(piece['right_eye_angle'])         # 5
                new_input_data.append(piece['right_shoulder_angle'])    # 6
                new_input_data.append(piece['left_shoulder_angle'])     # 7
                new_input_data.append(piece['nose_vec_y'])              # 8
                new_input_data.append(piece['nose_vec_x'])              # 9
                new_input_data.append(piece['gaze'])                    # 10
                new_input_data.append(0)                                # book 11
                new_input_data.append(0)                                # bottle 12
                new_input_data.append(0)                                # bowl 13
                new_input_data.append(0)                                # cup 14
                new_input_data.append(0)                                # laptop 15
                new_input_data.append(0)                                # cell phone 16
                new_input_data.append(0)                                # blocks 17
                new_input_data.append(0)                                # tablet 18
                new_input_data.append(0)                                # unknown 19
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
                    elif item == 'block':
                        new_input_data[foi+6] += 1
                    elif item == 'tablet':
                        new_input_data[foi+7] += 1
                    else:
                        new_input_data[foi+8] += 1

                X.append(new_input_data)
                Y.append(label)

    X = np.array(X)
    Y = np.array(Y)
    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    colors = ['navy', 'darkorange']
    plt.figure(figsize=(8, 8))
    for color, i, target in zip(colors, [-1, 1], ['Uninterruptible', 'Interruptible']):
        plt.scatter(X_pca[Y==i, 0], X_pca[Y==i, 1], color=color, lw=2, label=target)
    plt.title('PCA of binary interruptibility dataset')
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    if scale:
        plt.axis([-4, 4, -4, 6])
    else:
        plt.axis([-80, 150, -150, 150])
    plt.show()

# vis_rel_data(scale=True, do_heatmap=False)
# unintstd, unintmean = get_stddvs(normalize=True, automated=False, label=0)
# intstd, intmean = get_stddvs(normalize=True, automated=False, label=1)
# allstd, allmean = get_stddvs(normalize=True, automated=False, label=2)
#
# unintstdiff = np.abs(allstd - unintstd)
# intstddiff = np.abs(allstd - intstd)
# unintmeandiff = np.abs(allmean - unintmean)
# intmeandiff = np.abs(allmean - intmean)
#
# diff_between_two = intstddiff - unintstdiff
# print diff_between_two*100
# diff_again = abs(intstd) - abs(unintstd)
# print diff_again*100
