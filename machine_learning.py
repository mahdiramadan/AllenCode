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

class MachineLearning:
    def __init__(self, exp_folder, lims_ID):

        self.ip = ip(exp_folder, lims_ID)
        self.ep = ep(exp_folder, lims_ID)

    def run_svm(self):

        input = self.get_data()

        clf = NuSVC( class_weight='auto')
        clf.fit(X, input['labeled_vector'])


    def get_data(self):

        hf = h5py.File('data.h5', 'r')
        final_data = []
        dimension = 260*540
        wheel = pickle.load(open('wheel.pickle', 'rb'))

        for item in range(1, 3):
            group = hf.get('first ' + str(item) + '000 frames')
            # optical.append(np.array(group.get('optical')))
            # angles.append(np.array(group.get('angles')))
            for f in range(len(np.array(group.get('frames')))):
                frame = np.array(group.get('frames'))[f]
                orientation = color.rgb2gray(frame)
                fd, hog_image = hog(orientation, orientations=8, pixels_per_cell=(10, 10), cells_per_block=(4, 4), visualise=True)
                hog_image = np.reshape(hog_image, (1, dimension))
                optical = np.reshape(np.array(group.get('optical'))[f], (1,dimension))
                angles = np.reshape(np.array(group.get('angles'))[f], (1,dimension))
                final_data.append(np.concatenate((hog_image,optical, angles), axis = 1))


        label = 'fidget'
        index = self.ep.get_labels().index(label) + 1
        labeled_vector = self.ep.get_per_frame_data()[index]

        return {'feature_data': final_data, 'label': labeled_vector}





