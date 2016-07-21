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
import cv2
import matplotlib.pyplot as plt
import numpy as np
import h5py

class MachineLearning:
    def __init__(self, exp_folder, lims_ID):

        self.ip = ip(exp_folder, lims_ID)
        self.ep = ep(exp_folder, lims_ID)

    def run_svm(self):

        input = self.get_data()






    def get_data(self):

        optical = pickle.load(open('optical.pickle', 'rb'))

        angle = pickle.load(open('angle.pickle', 'rb'))

        wheel = pickle.load(open('wheel.pickle', 'rb'))

        frames = pickle.load(open('frames.pickle', 'rb'))

        label = 'fidget'
        index = self.ep.get_labels().index(label) + 1

        labeled_vector = self.ep.get_per_frame_data()[index]

        return {'optical': optical, 'angle': angle, 'wheel': wheel, 'frames': frames, 'label': labeled_vector}





