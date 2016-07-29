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
import time
import math
from skimage import feature
from scipy import ndimage
from sklearn import neighbors, datasets

class MachineLearning:
    def __init__(self, exp_folder, lims_ID):

        self.ip = ip(exp_folder, lims_ID)
        self.ep = ep(exp_folder, lims_ID)

    def run_svm(self):

        input = self.get_data()

        # X_train = input['feature_data'][0:8000]
        # X_test = input['feature_data'][8000:12000]
        #
        # y_train = input['label'][0:8000]
        # y_test = input['label'][8000:12000]
        #
        #
        # # # Set the parameters by cross-validation
        # # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10, 1, 1e-3, 1e-4],
        # #                      'C': [0.001, 0.1, 1, 10, 100, 1000, 10000]},
        # #                     {'kernel': ['linear'], 'C': [0.001, -0.1, 1, 10, 100, 1000, 10000]}]
        # #
        # # scores = ['accuracy', 'f1']
        # #
        # # for score in scores:
        # #     print("# Tuning hyper-parameters for %s" % score)
        # #     print()
        # #
        # #     clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
        # #                        scoring='%s' % score)
        # #     clf.fit(X_train, y_train)
        # #
        # #     print("Best parameters set found on development set:")
        # #     print()
        # #     print(clf.best_params_)
        # #     print()
        # #     print("Grid scores on development set:")
        # #     print()
        # #     for params, mean_score, scores in clf.grid_scores_:
        # #         print("%0.3f (+/-%0.03f) for %r"
        # #               % (mean_score, scores.std() * 2, params))
        # #     print()
        # #
        # #     print("Detailed classification report:")
        # #     print()
        # #     print("The model is trained on the full development set.")
        # #     print("The scores are computed on the full evaluation set.")
        # #     print()
        # #     y_true, y_pred = y_test, clf.predict(X_test)
        # #     print(classification_report(y_true, y_pred))
        # #     print()
        #
        # # # clf = RandomForestClassifier(verbose=3)
        # # clf = SVC(kernel='linear', C = 0.1, verbose = 2)
        #
        # clf = neighbors.KNeighborsClassifier(2)
        # clf.fit(X_train, y_train)
        #
        # joblib.dump(clf, 'clf.pkl')
        #
        # y_true, y_pred = y_test, clf.predict(X_test)
        # print(classification_report(y_true, y_pred))

        # # clf = joblib.load('filename.pk1')

    def get_data(self):

        hf = h5py.File('data.h5', 'r')
        final_data = []
        dimension = 260*540
        wheel = pickle.load(open('wheel.pickle', 'rb'))
        wheel = (wheel - np.min(wheel)) / (np.max(wheel) - np.min(wheel) + 10 ** -10)


        count = 0
        k = 0
        first = 0
        second = 0
        hist_1 = []
        hist_2 = []
        hist_3 = []
        hist_4 = []
        hist_5 = []
        hist_6 = []


        # label = 'fidget'
        # index = self.ep.get_labels().index(label) + 1
        # labeled_vector = np.array(self.ep.get_per_frame_data()[index])
        #
        # label = 'walking'
        # index = self.ep.get_labels().index(label) + 1
        # walking= np.array(self.ep.get_per_frame_data()[index])
        #
        # label = 'running'
        # index = self.ep.get_labels().index(label) + 1
        # running = np.array(self.ep.get_per_frame_data()[index])

        for item in range(1, 2):
            group = hf.get('first ' + str(item) + '000 frames')
            # optical.append(np.array(group.get('optical')))
            # angles.append(np.array(group.get('angles')))
            frames = np.array(group.get('frames'))
            opticals = np.array(group.get('optical'))
            angles = np.array(group.get('angles'))
            count += 1

            print ('first ' + str(count) + '000 frames')

            # width = len(frames[0][0])
            # height = len(frames[0])
            # dim = height / 2 * (width / 2)

            hsv = np.zeros((260,540,3))
            hsv[..., 1] = 255

            for f in range(len(frames)):


                # hog = self.hog(frames[f])
                # hog =
                #  np.reshape(hog, (1, len(hog)))
                # hog = (hog - hog.min())/(hog.max() - hog.min() + 10**-1)

                # section_2 = frames[f][0:height / 3, width / 4:width]
                # height2= len(section_2)
                # width2= len(section_2[0])
                #
                # section_3 = frames[f][height/3: 2*height/3, 0: width/2]
                # height3 = len(section_3)
                # width3 = len(section_3[0])
                #
                # section_4 = frames[f][height/3: height, width/2: width]
                # height4= len(section_4)
                # width4= len(section_4[0])

                # hist_1, bin_1 = np.histogram(np.reshape((opticals[f][height / 2: height, 0: width / 2]), (1, dim)), 20)
                # b_width = 0.7 * (bin_1[1] - bin_1[0])
                # center = (bin_1[:-1] + bin_1[1:]) / 2
                # plt.bar(center, hist_1, align='center', width=b_width)
                # plt.show()
                #
                # hist_1, bin_1 = np.histogram(np.reshape((opticals[f][0:height / 2, width / 2:width]), (1, dim)), 20)
                # b_width = 0.7 * (bin_1[1] - bin_1[0])
                # center = (bin_1[:-1] + bin_1[1:]) / 2
                # plt.bar(center, hist_1, align='center', width= b_width)
                # plt.show()
                #
                # hist_1, bin_1 = np.histogram(np.reshape((opticals[f][height / 2:height, width / 2: width]), (1, dim)), 20)
                # b_width = 0.7 * (bin_1[1] - bin_1[0])
                # center = (bin_1[:-1] + bin_1[1:]) / 2
                # plt.bar(center, hist_1, align='center', width= b_width)
                # plt.show()
                #
                #
                # optical = np.reshape(optical, (1, len(optical)))
                # optical = (optical - optical.min()) / (optical.max() - optical.min() + 10**-10)
                #
                # angle = self.hog(angles[f])
                # angle = np.reshape(angle, (1, len(angle)))
                # angle = (angle - angle.min()) / (angle.max() - angle.min()+ 10**-10)
                #
                # vector = np.int16(np.hstack(( optical, angle)))

                hsv[..., 0] = angles[f]
                hsv[..., 2] = cv2.normalize(opticals[f], None, 0, 255, cv2.NORM_MINMAX)
                hsv = np.float32(hsv)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                r = np.int64(rgb[:,:,0])
                # x = x[np.nonzero(x)]
                b = np.int64(rgb[:,:,1])
                g = np.int64(rgb[:,:,2])

                final_data = []
                frame_data = []
                vector = []
                optic1 = []
                optic2 = []
                optic3= []

                # for (x,y,window) in self.sliding_window(frames[f], 60, (60,60)):
                #     yy = np.bincount(np.squeeze(np.reshape(window, (1, 3600))))
                #     ii = np.nonzero(yy)[0]
                #     hist_x = np.multiply(ii, yy[ii])
                #     frame_data = np.concatenate((frame_data, hist_x))

                for (x, y, window) in self.sliding_window(r, 60, (60, 60)):
                    hist, bin = np.histogram(r, 20)
                    center = (bin[:-1] + bin[1:]) / 2
                    yy = np.bincount(np.squeeze(np.reshape(window, (1, 3600))))
                    ii = np.nonzero(yy)[0]
                    hist_x = np.multiply(ii, yy[ii])
                    optic1 = np.concatenate((optic1, hist_x))

                for (x, y, window) in self.sliding_window(b, 60, (60, 60)):
                    yy = np.bincount(np.squeeze(np.reshape(window, (1, 3600))))
                    ii = np.nonzero(yy)[0]
                    hist_x = np.multiply(ii, yy[ii])
                    optic2 = np.concatenate((optic2, hist_x))

                for (x, y, window) in self.sliding_window(g, 60, (60, 60)):

                    yy = np.bincount(np.squeeze(np.reshape(window, (1, 3600))))
                    ii = np.nonzero(yy)[0]
                    hist_x = np.multiply(ii, yy[ii])
                    optic3 = np.concatenate((optic3, hist_x))

                if count == 1:
                    final_data = np.concatenate((frame_data, optic1, optic2, optic3))

                else:
                    vector = np.concatenate((frame_data, optic1, optic2, optic3))


            final_data = np.vstack((final_data, vector))



        return {'feature_data': final_data, 'label': labeled_vector}

    def hog(self, img):

        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        return mag

    def show_frame(self, frame):
        cv2.imshow('image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def sliding_window(self, image, stepSize, windowSize):
        # slide a window across the image
        for y in xrange(0, image.shape[0] - stepSize, stepSize):
            for x in xrange(0, image.shape[1]- stepSize, stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


