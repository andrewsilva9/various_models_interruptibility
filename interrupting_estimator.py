from __future__ import division
import pickle
import json
import os
import math
import sys
import itertools
import numpy as np
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import rospy
from rail_cpm.msg import Keypoints, Poses

# Debug Helpers
FAIL_COLOR = '\033[91m'
ENDC_COLOR = '\033[0m'


def eprint(error):
    sys.stderr.write(
        FAIL_COLOR
        + type(error).__name__
        + ": "
        + error.message
        + ENDC_COLOR
    )
# End Debug Helpers


class MLPEstimator(object):
    """
    This class takes in image data and finds / annotates objects within the image
    """

    def __init__(self):
        rospy.init_node('interrupt_estimate')
        self.clf = joblib.load('MLPrel.pkl')
        self.scaler = joblib.load('scaler.pkl')
        self.debug = rospy.get_param('~debug', default=False)
        self.feature_vector_sub_topic_name = rospy.get_param('~feature_vector_sub_topic_name',
                                                             default='/data_filter/feature_vector')

    def make_estimation(self, feature_vector):
        """
        Take feature vectors and run them through the MLP classifier
        """

        header = feature_vector.header
        x_in = []
        x_in.append(feature_vector.pose.right_wrist_angle)                # 0
        x_in.append(feature_vector.pose.right_elbow_angle)                # 1
        x_in.append(feature_vector.pose.left_wrist_angle)                 # 2
        x_in.append(feature_vector.pose.left_elbow_angle)                 # 3
        x_in.append(feature_vector.pose.left_eye_angle)                   # 4
        x_in.append(feature_vector.pose.right_eye_angle)                  # 5
        x_in.append(feature_vector.pose.right_shoulder_angle)             # 6
        x_in.append(feature_vector.pose.left_shoulder_angle)              # 7
        x_in.append(feature_vector.pose.nose_vec_y)                       # 8
        x_in.append(feature_vector.pose.nose_vec_x)                       # 9

        x_in.append(feature_vector.gaze_boolean.gaze_at_robot)            # 10

        x_in.append(0)  # book 11
        x_in.append(0)  # bottle 12
        x_in.append(0)  # bowl 13
        x_in.append(0)  # cup 14
        x_in.append(0)  # laptop 15
        x_in.append(0)  # cell phone 16
        x_in.append(0)  # blocks 17
        x_in.append(0)  # tablet 18
        x_in.append(0)  # unknown 19
        foi = 11  # first object index to make it easier to change stuff
        for item in feature_vector.object_labels.object_labels:
            if item == 'book':
                x_in[foi] += 1
            elif item == 'bottle':
                x_in[foi + 1] += 1
            elif item == 'bowl':
                x_in[foi + 2] += 1
            elif item == 'cup':
                x_in[foi + 3] += 1
            elif item == 'laptop':
                x_in[foi + 4] += 1
            elif item == 'cell phone':
                x_in[foi + 5] += 1
            elif item == 'blocks':
                x_in[foi + 6] += 1
            elif item == 'tablet':
                x_in[foi + 7] += 1
            else:
                x_in[foi + 8] += 1

        x_in = self.scaler.transform(x_in)
        pred = self.clf.predict(x_in)

        msg = outgoing_message_type()
        msg.header = header
        msg.person_id = feature_vector.person_id
        msg.prediction = pred

        self.pred_pub.publish(msg)

    def run(self,
            pub_object_topic='~predictions'):
        rospy.Subscriber(self.feature_vector_sub_topic_name, FeatureVector, self.make_estimation) # subscribe to sub_image_topic and callback parse
        self.pred_pub = rospy.Publisher(pub_object_topic, Poses, queue_size=2) # objects publisher
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = CPM()
        detector.run()
    except rospy.ROSInterruptException:
        pass
