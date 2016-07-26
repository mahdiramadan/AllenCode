"""machine_learning.py by Mahdi Ramadan, 06-18-2016
This program will be used for machine learning fitting
and prediction
"""
import os
import pandas
import sys
from image_processing import ImageProcessing as ip
from excel_processing import ExcelProcessing as ep
import pickle
from sklearn.svm import NuSVC
import cv2
import matplotlib.pyplot as plt
import numpy as np
import h5py
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.externals import joblib
from itertools import product
from math import floor, pi
from scipy.ndimage.measurements import sum as ndi_sum
import time

class MachineLearning:
    def __init__(self, exp_folder, lims_ID):

        self.ip = ip(exp_folder, lims_ID)
        self.ep = ep(exp_folder, lims_ID)

    def run_svm(self):

        input = self.get_data()

        X_train = preprocessing.StandardScaler().fit(input['feature_data'][0:50]).transform(input['feature_data'][0:50])
        X_test = preprocessing.StandardScaler().fit(input['feature_data'][50:100]).transform(input['feature_data'][50:100])

        y_train = input['label'][0:50]
        y_test = input['label'][50:100]


        # # Set the parameters by cross-validation
        # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5],
        #                      'C': [0.1, 1, 10, 100, 1000]},
        #                     {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]}]
        #
        # scores = ['precision', 'recall']
        #
        # for score in scores:
        #     print("# Tuning hyper-parameters for %s" % score)
        #     print()
        #
        #     clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=4,
        #                        scoring='%s_weighted' % score)
        #     clf.fit(X_train, y_train)
        #
        #     print("Best parameters set found on development set:")
        #     print()
        #     print(clf.best_params_)
        #     print()
        #     print("Grid scores on development set:")
        #     print()
        #     for params, mean_score, scores in clf.grid_scores_:
        #         print("%0.3f (+/-%0.03f) for %r"
        #               % (mean_score, scores.std() * 2, params))
        #     print()
        #
        #     print("Detailed classification report:")
        #     print()
        #     print("The model is trained on the full development set.")
        #     print("The scores are computed on the full evaluation set.")
        #     print()
        #     y_true, y_pred = y_test, clf.predict(X_test)
        #     print(classification_report(y_true, y_pred))
        #     print()

        clf = SVC(kernel='linear', C=0.1, class_weight='auto')
        clf.fit(X_train, y_train)

        joblib.dump(clf, 'clf.pkl')

        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))

        # clf = joblib.load('filename.pk1')

    def get_data(self):

        hf = h5py.File('data.h5', 'r')
        final_data = []
        dimension = 260*540
        wheel = pickle.load(open('wheel.pickle', 'rb'))
        # wheel = preprocessing.StandardScaler().fit(wheel).transform(wheel)

        for item in range(1, 2):
            group = hf.get('first ' + str(item) + '000 frames')
            # optical.append(np.array(group.get('optical')))
            # angles.append(np.array(group.get('angles')))
            for f in range(500, 520):

                hog = cv2.HOGDescriptor()
                h = hog.compute(np.array(group.get('frames'))[f])
                h2 = hog.compute(np.array(group.get('optical'))[f])
                angles = np.reshape(np.array(group.get('angles'))[f], (1,dimension))
                final_data.append(np.concatenate((h, h2, angles), axis = 1))

        final_data = np.vstack(final_data)

        label = 'fidget'
        index = self.ep.get_labels().index(label) + 1
        labeled_vector = np.array(self.ep.get_per_frame_data()[index])

        return {'feature_data': final_data, 'label': labeled_vector[500:520]}

    # def findHOGFeaturesVect(self, img, n_divs=6, n_bins=6):
    #     """
    #     **SUMMARY**
    #     Get HOG(Histogram of Oriented Gradients) features from the image.
    #
    #
    #     **PARAMETERS**
    #     * *img*    - SimpleCV.Image instance
    #     * *n_divs* - the number of divisions(cells).
    #     * *n_divs* - the number of orientation bins.
    #
    #     **RETURNS**
    #     Returns the HOG vector in a numpy array
    #
    #     """
    #     # Size of HOG vector
    #     n_HOG = n_divs * n_divs * n_bins
    #
    #     # Initialize output HOG vector
    #     # HOG = [0.0]*n_HOG
    #     HOG = np.zeros((n_HOG, 1))
    #
    #     # Apply sobel on image to find x and y orientations of the image
    #     Ix = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    #     Iy = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
    #
    #     height = len(img)
    #     width = len(img[0])
    #
    #     # Area of image
    #     img_area = height * width
    #
    #     # Range of each bin
    #     BIN_RANGE = (2 * pi) / n_bins
    #
    #     # m = 0
    #     angles = np.arctan2(Iy, Ix)
    #     magnit = ((Ix ** 2) + (Iy ** 2)) ** 0.5
    #
    #     bins = np.int16((angles[..., 0] % (2 * pi) / BIN_RANGE))
    #     y, x = np.meshgrid(np.arange(height), np.arange(width))
    #     x = np.int16(x / width * n_divs)
    #     y = np.int16(y / height * n_divs)
    #     labels = (x * n_divs + y) * n_bins + bins
    #     index = np.arange(n_HOG)
    #     HOG = ndi_sum(magnit[..., 0], labels, index)
    #
    #     return HOG / img_area