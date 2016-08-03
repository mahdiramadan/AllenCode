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
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D


class ClusteringVisualization:
    def __init__(self, exp_folder, lims_ID):

        self.ip = ip(exp_folder, lims_ID)
        self.ep = ep(exp_folder, lims_ID)

    def get_cluster(self):

        input = self.get_data()

        # X_train = input['feature_data'][0:8000]
        # X_test = input['feature_data'][8000:12000]
        #
        # y_train = input['label'][0:8000]
        # y_test = input['label'][8000:12000]
        #
        #
        # # # Set the parameters by cross-validation
        # # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5],
        # #                      'C': [0.1, 1, 10, 100, 1000]},
        # #                     {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]}]
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
        # wheel = preprocessing.StandardScaler().fit(wheel).transform(wheel)
        count = 0
        k = 0
        first = 0
        second = 0
        hist_1 = []
        hist_0 = []
        hist_2 = []
        hist_3 = []
        hist_4 = []
        hist_5 = []
        label = 'fidget'
        index = self.ep.get_labels().index(label) + 1
        labeled_vector = np.array(self.ep.get_per_frame_data()[index])

        label = 'walking'
        index = self.ep.get_labels().index(label) + 1
        walking = np.array(self.ep.get_per_frame_data()[index])

        label = 'running'
        index = self.ep.get_labels().index(label) + 1
        running = np.array(self.ep.get_per_frame_data()[index])

        movement_vector = []
        movement_vector.append([sum(x) for x in zip(walking, running)])

        questionable_frames = []

        for item in range(1,51):
            group = hf.get('first ' + str(item) + '000 frames')
            # optical.append(np.array(group.get('optical')))
            # angles.append(np.array(group.get('angles')))
            frames = np.array(group.get('frames'))
            opticals = np.array(group.get('optical'))
            angles = np.array(group.get('angles'))
            count += 1
            print ('first ' + str(count) + '000 frames')

            width = len(frames[0][0])
            height = len(frames[0])
            dim = height / 2 * (width / 2)

            for f in range(520,525):


                # hog = self.hog(frames[f])
                # hog =
                #  np.reshape(hog, (1, len(hog)))
                # hog = (hog - hog.min())/(hog.max() - hog.min() + 10**-1)

                #
                # optical_1 = self.hog(section_1)
                # optical_2 = self.hog(section_2)
                # optical_3 = self.hog(section_3)
                # optical_4 = self.hog(section_4)

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
                #
                if movement_vector[0][f] == 1:
                    # if np.mean(opticals[f]) < 4:
                    #     questionable_frames.append(k)
                    first += 1
                    if first == 1:
                        hist_1 = np.mean(opticals[f])
                        hist_2 = self.get_mode(angles[f])
                        hist_4 = np.mean(frames[f])
                    else:
                        hist_1 = np.hstack((hist_1, np.mean(opticals[f])))
                        hist_2 = np.hstack((hist_2, self.get_mode(angles[f])))
                        hist_4 = np.hstack((hist_4, np.mean(frames[f])))

                if labeled_vector[f] == 1:
                    # if np.mean(opticals[f]) > 4:
                    #     questionable_frames.append(k)

                    second += 1
                    if second == 1:
                        hist_0 = np.mean(opticals[f])
                        hist_3 = self.get_mode(angles[f])
                        hist_5 = np.mean(frames[f])
                    else:
                        hist_0 = np.hstack((hist_0, np.mean(opticals[f])))
                        hist_3 = np.hstack((hist_3, self.get_mode(angles[f])))
                        hist_5 = np.hstack((hist_5, np.mean(frames[f])))

                k += 1



                # final_data.append(preprocessing.StandardScaler().fit(vector).transform(vector))

        # hist_0 = joblib.load('hist_0.pkl')
        # hist_1 = joblib.load('hist_1.pkl')
        # hist_2 = joblib.load('hist_2.pkl')
        # hist_3 = joblib.load('hist_3.pkl')
        # hist_4 = joblib.load('hist_4.pkl')
        # hist_5 = joblib.load('hist_5.pkl')

        print(questionable_frames)


        x = np.mean(hist_1)
        y = np.std(hist_1)
        z = np.max(hist_1)
        print(x,y,z)

        fig = plt.figure(1)
        ax = Axes3D(fig)
        # hist_1, bin_1 = np.histogram(hist_1, 20, normed= True)
        # b_width = 0.7 * (bin_1[1] - bin_1[0])
        # center = (bin_1[:-1] + bin_1[1:]) / 2
        ax.plot(hist_1, hist_2, hist_4, 'bo')


        x = np.mean(hist_0)
        y = np.std(hist_0)
        z = np.max(hist_0)

        print(x, y, z)

        # hist_0, bin_0 = np.histogram(hist_0, 20, normed= True)
        # b_width = 0.7 * (bin_1[1] - bin_1[0])
        # center = (bin_1[:-1] + bin_1[1:]) / 2

        ax.plot(hist_0, hist_3, hist_5, 'go')
        ax.azim = 200
        ax.elev = -45
        ax.set_xlabel('Optical flow')
        ax.set_ylabel('Optical Angle')
        ax.set_zlabel('Frame HOG')
        plt.show()



        # return {'feature_data': final_data, 'label': labeled_vector}

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
        return mag

    def show_frame(self, frame):
        cv2.imshow('image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def get_features(self, image, bins=8, angle=360., pyramid_levels=3):
        """
        Returns a feature vector containing a PHOG descriptor of a whole image.
        :param image_path: Absolute path to an image
        :param bins: Number of (orientation) bins on the histogram (optimal: 20)
        :param angle: 180 or 360 (optimal: 360)
        :param pyramid_levels: Number of pyramid levels (optimal: 3)
        :return:
        """

        feature_vec = self.phog(image, bins, angle, pyramid_levels)
        feature_vec = feature_vec.T[0]  # Transpose vector, take the first array
        return feature_vec

    def phog(self, image, bin, angle, pyramid_levels):
        """
        Given and image I, phog computes the Pyramid Histogram of Oriented
        Gradients over L pyramid levels and over a Region Of Interest.
        :param image_path: Absolute path to an image of size MxN (Color or Gray)
        :param bin: Number of (orientation) bins on the histogram
        :param angle: 180 or 360
        :param pyramid_levels: Number of pyramid levels
        :return: Pyramid histogram of oriented gradients
        """

        grayscale_img = image  # 0 converts it to grayscale
        x = image.shape[0]
        y= image.shape[1]

        bh = np.array([])
        bv = np.array([])
        if np.sum(np.sum(grayscale_img)) > 100.:
            # Matlab The default sigma is sqrt(2); the size of the filter is chosen automatically, based on sigma.
            # Threshold is applied automatically - the percentage is a bit different than in Matlab's implementation:
            # low_threshold: 10%
            # high_threshold: 20%
            edges_canny = feature.canny(grayscale_img, sigma=math.sqrt(2))
            [GradientY, GradientX] = np.gradient(np.double(grayscale_img))
            GradientYY = np.gradient(GradientY)[1]  # Take only the first matrix
            Gr = np.sqrt((GradientX * GradientX + GradientY * GradientY))

            index = GradientX == 0.
            index = np.int16(index)  # Convert boolean array to an int array
            GradientX[np.where(index > 0)] = np.power(10, -5)
            YX = GradientY / GradientX

            if angle == 180.:
                angle_values = np.divide((np.arctan(YX) + np.pi / 2.) * 180., np.pi)
            if angle == 360:
                angle_values = np.divide((np.arctan2(GradientY, GradientX) + np.pi) * 180., np.pi)

            [bh, bv] = self.bin_matrix(angle_values, edges_canny, Gr, angle, bin)
        else:
            bh = np.zeros((x,y))
            bv = np.zeros((x,y))

        # Don't consider a roi, take the whole image instead
        bh_roi = bh
        bv_roi = bv
        p = self.phog_descriptor(bh_roi, bv_roi, pyramid_levels, bin)

        return p

    def bin_matrix(self, angle_values, edge_image, gradient_values, angle, bin):
        """
        Computes a Matrix (bm) with the same size of the image where
        (i,j) position contains the histogram value for the pixel at position (i,j)
        and another matrix (bv) where the position (i,j) contains the gradient
        value for the pixel at position (i,j)
        :param angle_values: Matrix containing the angle values
        :param edge_image: Edge Image
        :param gradient_values: Matrix containing the gradient values
        :param angle: 180 or 360
        :param bin: Number of bins on the histogram
        :return: bm - Matrix with the histogram values
                bv - Matrix with the gradient values (only for the pixels belonging to and edge)
        """

        # 8-orientations/connectivity structure (Matlab's default is 8 for bwlabel)
        structure_8 = [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]

        [contorns, n] = ndimage.label(edge_image, structure_8)
        X = edge_image.shape[1]
        Y = edge_image.shape[0]
        bm = np.zeros((Y, X))
        bv = np.zeros((Y, X))
        nAngle = np.divide(angle, bin)
        for i in np.arange(1, n + 1):
            [posY, posX] = np.nonzero(contorns == i)
            posY = posY + 1
            posX = posX + 1
            for j in np.arange(1, (posY.shape[0]) + 1):
                pos_x = posX[int(j) - 1]
                pos_y = posY[int(j) - 1]
                b = np.ceil(np.divide(angle_values[int(pos_y) - 1, int(pos_x) - 1], nAngle))
                if b == 0.:
                    bin = 1.
                if gradient_values[int(pos_y) - 1, int(pos_x) - 1] > 0:
                    bm[int(pos_y) - 1, int(pos_x) - 1] = b
                    bv[int(pos_y) - 1, int(pos_x) - 1] = gradient_values[int(pos_y) - 1, int(pos_x) - 1]

        return [bm, bv]

    def phog_descriptor(self, bh, bv, pyramid_levels, bin):
        """
        Computes Pyramid Histogram of Oriented Gradient over an image.
        :param bh: Matrix of bin histogram values
        :param bv: Matrix of gradient values
        :param pyramid_levels: Number of pyramid levels
        :param bin: Number of bins
        :return: Pyramid histogram of oriented gradients (phog descriptor)
        """

        p = np.empty((0, 1), dtype=int)  # dtype=np.float64? # vertical size 0, horizontal 1

        for b in np.arange(1, bin + 1):
            ind = bh == b
            ind = np.int16(ind)  # convert boolean array to int array
            sum_ind = np.sum(bv[np.where(ind > 0)])
            p = np.append(p, np.array([[sum_ind]]), axis=0)  # append the sum horizontally to empty p array

        cella = 1.
        for l in np.arange(1, pyramid_levels + 1):  # defines a range (from, to, step)
            x = np.fix(np.divide(bh.shape[1], 2. ** l))
            y = np.fix(np.divide(bh.shape[0], 2. ** l))
            xx = 0.
            yy = 0.
            while xx + x <= bh.shape[1]:
                while yy + y <= bh.shape[0]:
                    bh_cella = np.array([])
                    bv_cella = np.array([])
                    bh_cella = bh[int(yy + 1.) - 1:yy + y, int(xx + 1.) - 1:xx + x]
                    bv_cella = bv[int(yy + 1.) - 1:yy + y, int(xx + 1.) - 1:xx + x]

                    for b in np.arange(1, bin + 1):
                        ind = bh_cella == b
                        ind = np.int16(ind) # convert boolean array to int array
                        sum_ind = np.sum(bv_cella[np.where(ind > 0)])
                        p = np.append(p, np.array([[sum_ind]]), axis=0)  # append the sum horizontally to p

                    yy = yy + y

                cella = cella + 1.
                yy = 0.
                xx = xx + x

        if np.sum(p) != 0:
            p = np.divide(p, np.sum(p))

        return p

    def get_mode(self, array):

        u, indices = np.unique(array, return_inverse=True)
        return u[np.argmax(np.bincount(indices))]

