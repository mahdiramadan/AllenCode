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
from sklearn.ensemble import RandomForestClassifier
from itertools import product
from math import floor, pi
from scipy.ndimage.measurements import sum as ndi_sum
import time
from sklearn.decomposition import PCA, IncrementalPCA

class MachineLearning:
    def __init__(self, exp_folder, lims_ID):

        self.ip = ip(exp_folder, lims_ID)
        self.ep = ep(exp_folder, lims_ID)

    def run_svm(self):

        input = self.get_data()

        X_train = input['feature_data'][0:8000]
        X_test = input['feature_data'][8000:10000]

        y_train = np.reshape(input['label'][0:8000], (8000, 1))
        y_test = np.reshape(input['label'][8000:10000], (2000,1))


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

        clf = RandomForestClassifier(verbose=3)
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
        count = 0

        for item in range(1, 11):
            group = hf.get('first ' + str(item) + '000 frames')
            # optical.append(np.array(group.get('optical')))
            # angles.append(np.array(group.get('angles')))
            frames = np.array(group.get('frames'))
            opticals = np.array(group.get('optical'))
            angles = np.array(group.get('angles'))
            count += 1
            print ('first ' + str(count) + '000 frames')


            for f in range(len(frames)):

                hog = self.hog(frames[f])
                hog = np.reshape(hog, (1, len(hog)))
                hog = 100 * (hog - hog.mean()) / hog.std()

                optical = self.hog(opticals[f])
                optical = np.reshape(optical, (1, len(optical)))
                optical = 100*(optical - optical.mean())/optical.std()

                angle = self.hog(angles[f])
                angle = np.reshape(angle, (1, len(angle)))
                optical = 100 * (angle - angle.mean()) / angle.std()

                vector = np.int16(np.hstack((hog, optical, angle)))

                if count == 1 and f == 0:
                    final_data = vector
                else:
                    final_data = np.vstack((final_data, vector))

                # final_data.append(preprocessing.StandardScaler().fit(vector).transform(vector))

        final_data = np.vstack(final_data)

        label = 'fidget'
        index = self.ep.get_labels().index(label) + 1
        labeled_vector = np.array(self.ep.get_per_frame_data()[index])

        return {'feature_data': final_data, 'label': labeled_vector}

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

    def hog(self, img):

        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)

        # quantizing binvalues in (0...16)
        bins = np.int32(16 * ang / (2 * np.pi))

        # Divide to 4 sub-squares
        bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), 16) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)
        return hist

    def show_frame(self, frame):
        cv2.imshow('image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()