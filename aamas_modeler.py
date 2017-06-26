import pickle
import json
import os
import math
import sys
import itertools
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib


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


def annotation_printer(annotation):
    for index, person in enumerate(annotation):
        for piece in person['data']:
            print index, piece['scene_object']
        # print annotation[person]
    print ''


def train_aamas(model_name, clf):
    annots_dir = '/Users/andrewsilva/aamas_annotations'
    X = []
    Y = []
    # TODO: Determine which data need to be in 'bad_keys'
    # bad_keys = ['body_bb_height', 'body_b', 'face_bb_height', 'body_d', 'body_c', 'base_is_moving', 'body_a',
    #             'body_z', 'body_bb_y', 'body_bb_x', 'audio_confidence', 'face_bb_width', 'face_landmarks_present',
    #             'timestamp', 'scene_object', 'name', 'body_bb_width', 'audio_angle', 'face_bb_x', 'face_bb_y',
    #             'face_is_frontal']
    bad_keys = ['timestamp', 'name']
    for filename in os.listdir(annots_dir):
        af = open(os.path.join(annots_dir, filename), 'rb')
        annotation = pickle.load(af)
        # For each person in the segmentation
        # annotation_printer(annotation)
        # return
        for person in annotation:
            # sorted_ann = sorted(annotation[person], key=lambda x: x['timestamp'])
            label = person['value']
            for piece in person['data']:
                new_input_data = []
                for key, value in piece.iteritems():
                    if key in bad_keys:
                        continue
                    if value is None:
                        piece[key] = -1.0
                    elif value is False:
                        piece[key] = 0.0
                    elif value is True:
                        piece[key] = 1.0
                # TODO: Change these keys to correspond to aamas_annotations data keys
                new_input_data.append(piece['body_y'])
                new_input_data.append(piece['body_x'])
                new_input_data.append(piece['face_is_frontal'])
                new_input_data.append(piece['body_a'])
                new_input_data.append(piece['body_b'])
                new_input_data.append(piece['body_c'])
                new_input_data.append(piece['body_d'])
                new_input_data.append(piece['audio_angle'])
                new_input_data.append(piece['audio_confidence'])
                new_input_data.append(piece['body_bb_height'])
                new_input_data.append(piece['body_bb_width'])
                new_input_data.append(piece['body_bb_y'])
                new_input_data.append(piece['body_bb_x'])
                new_input_data.append(piece['face_bb_height'])
                new_input_data.append(piece['face_bb_width'])
                new_input_data.append(piece['face_bb_x'])
                new_input_data.append(piece['face_bb_y'])
                new_input_data.append(piece['face_landmarks_present'])
                # TODO: Change how I'm handing 'scene_object' in aamas_annotations data.
                new_input_data.append(0)  # unknown 18
                new_input_data.append(0)  # none 19
                new_input_data.append(0)  # laptop 20
                new_input_data.append(0)  # bottle 21
                new_input_data.append(0)  # book 22
                new_input_data.append(0)  # headphones 23
                new_input_data.append(0)  # mug 24
                new_input_data.append(0)  # phone_talk 25
                new_input_data.append(0)  # phone_text 26
                item = piece['scene_object']
                if item == 'unknown':
                    new_input_data[18] += 1
                elif item == 'none':
                    new_input_data[19] += 1
                elif item == 'laptop':
                    new_input_data[20] += 1
                elif item == 'bottle':
                    new_input_data[21] += 1
                elif item == 'book':
                    new_input_data[22] += 1
                elif item == 'headphones':
                    new_input_data[23] += 1
                elif item == 'mug':
                    new_input_data[24] += 1
                elif item == 'phone_talk':
                    new_input_data[25] += 1
                elif item == 'phone_text':
                    new_input_data[26] += 1
                X.append(new_input_data)
                Y.append(label)

    class_names = ['0', '1', '2', '3', '4']
    kf = KFold(n_splits=10)
    X = np.array(X)
    Y = np.array(Y)
    # for train_index, test_index in kf.split(X):
    #     X_train, X_test = X[train_index].copy(), X[test_index].copy()
    #     y_train, y_test = Y[train_index].copy(), Y[test_index].copy()
    #     scaler = StandardScaler()
    #     scaler.fit(X_train)
    #     X_train = scaler.transform(X_train)
    #     X_test = scaler.transform(X_test)
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_test)
    #     cnf_matrix = confusion_matrix(y_test, y_pred)
    #     np.set_printoptions(precision=2)
    #     plt.figure()
    #     plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                           title='Normalized confusion matrix')
    #
    #     plt.show()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=69)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf.fit(X_train, y_train)
    joblib.dump(clf, model_name)
    y_pred = clf.predict(X_test)

    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
    return scaler


def extract_matlab_info(data_file, label_file):
    X = []
    Y = []
    use_min_features = True
    use_std_features = True
    use_ext_features = True
    use_object_features = True
    face_bool_feats = np.arange(0, 2)
    face_bb_features = np.arange(2, 6)
    body_bb_features = np.arange(6, 10)
    body_xyz_features = np.arange(10, 13)
    body_quat_features = np.arange(13, 17)
    audio_features = np.arange(17, 19)
    object_OR_features = 21
    object_bool_features = np.arange(21, 30)

    body_bb_derived_features = 42
    body_vel_derived_features = np.arange(32, 34)
    body_xyz_derived_features = [31, 40, 41]
    body_avel_derived_features = np.arange(35, 38)
    audio_derived_features = 39
    for index, seq in enumerate(data_file['sequences'][0]):
        for element in seq:
            input_arr = []
            usable_feats = []
            if use_min_features:
                usable_feats.append(element[face_bool_feats][:])
                usable_feats.append(element[body_xyz_features][:])
            if use_std_features:
                usable_feats.append(element[body_quat_features][:])
                usable_feats.append(element[audio_features][:])
            if use_ext_features:
                usable_feats.append(element[audio_derived_features])
                usable_feats.append(element[body_xyz_derived_features])
                usable_feats.append(element[body_vel_derived_features])
                usable_feats.append(element[body_avel_derived_features])
                usable_feats.append(element[body_bb_derived_features])
                usable_feats.append(element[body_bb_features])
                usable_feats.append(element[face_bb_features])
            if use_object_features:
                usable_feats.append(element[object_bool_features])
            usable_feats = np.array(usable_feats)
            usable_feats = usable_feats.flatten()
            iters = []
            for bit in usable_feats:
                if isinstance(bit, float):
                    iters.append(bit)
                else:
                    for b in bit:
                        iters.append(b)
            for piece in iters:
                if math.isnan(piece):
                    input_arr.append(-1)
                else:
                    input_arr.append(piece)
            X.append(input_arr)
    for index, seq in enumerate(label_file['labels'][0]):
        for element in seq[0]:
            Y.append(element)
    return X, Y

def train_full_aamas(model_name, clf):
    annots_dir = '/Users/andrewsilva/aamasmatlab'
    X = []
    Y = []
    # TODO: Determine which data need to be in 'bad_keys'
    # bad_keys = ['body_bb_height', 'body_b', 'face_bb_height', 'body_d', 'body_c', 'base_is_moving', 'body_a',
    #             'body_z', 'body_bb_y', 'body_bb_x', 'audio_confidence', 'face_bb_width', 'face_landmarks_present',
    #             'timestamp', 'scene_object', 'name', 'body_bb_width', 'audio_angle', 'face_bb_x', 'face_bb_y',
    #             'face_is_frontal']
    bad_keys = ['timestamp', 'name']
    data = sio.loadmat(os.path.join(annots_dir, 'data.mat'))
    labels = sio.loadmat(os.path.join(annots_dir, 'labels.mat'))
    X, Y = extract_matlab_info(data, labels)
    class_names = ['0', '1', '2', '3', '4']
    # kf = KFold(n_splits=10)
    X = np.array(X)
    Y = np.array(Y)
    # for train_index, test_index in kf.split(X):
    #     X_train, X_test = X[train_index].copy(), X[test_index].copy()
    #     y_train, y_test = Y[train_index].copy(), Y[test_index].copy()
    #     scaler = StandardScaler()
    #     scaler.fit(X_train)
    #     X_train = scaler.transform(X_train)
    #     X_test = scaler.transform(X_test)
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_test)
    #     cnf_matrix = confusion_matrix(y_test, y_pred)
    #     np.set_printoptions(precision=2)
    #     plt.figure()
    #     plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                           title='Normalized confusion matrix')
    #
    #     plt.show()
    X = np.array(X)
    Y = np.array(Y)
    kf = KFold(n_splits=10, shuffle=False)
    fold_count = 1
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
        y_train, y_test = Y[train_idx].copy(), Y[test_idx].copy()
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print 'Fold:', fold_count
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')

        plt.show()
        fold_count += 1
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=69)
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    #
    # clf.fit(X_train, y_train)
    # joblib.dump(clf, model_name)
    # y_pred = clf.predict(X_test)


    return scaler


if __name__ == '__main__':
    if len(sys.argv) > 1:
        adir = sys.argv[1]
    # change_to_relations()
    # plot_angles()
    model_name = 'aamas_matlab_random_forest_min_obj.pkl'
    use_prior = False
    clf = RandomForestClassifier(class_weight="balanced_subsample", n_estimators=10)
    # clf = MLPClassifier(max_iter=40000, tol=1e-4)
    # clf = KNeighborsClassifier()
    scaler = train_full_aamas(model_name, clf)
    # test_temporal_model_rel(scaler, model_name, use_prior)
