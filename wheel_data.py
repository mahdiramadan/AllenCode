"""wheel_data.py by Mahdi Ramadan, 07-12-2016
This program will be used for wheel data processing
and visualization
"""
import os
from stimulus_behavior import StimulusBehavior as sb
from excel_processing import ExcelProcessing as ep
from synced_videos import SyncedVideos as sv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import pickle
import time
from sklearn.decomposition import PCA
import scipy.optimize as optimization



class WheelData:
    def __init__(self, exp_folder, lims_ID):

        for file in os.listdir(exp_folder):
            # looks for the excel file and makes the directory to it
            if file.endswith(".mp4") and file.startswith(lims_ID):
                self.directory = exp_folder
                self.file_string = os.path.join(exp_folder, file)
                self.sb = sb(exp_folder)
                self.ep = ep(exp_folder, lims_ID)
                self.sv = sv (exp_folder, lims_ID)
                self.video_pointer = cv2.VideoCapture(self.file_string)

        if os.path.isfile(self.file_string):
                self.data_present = True
        else:
                self.data_present = False

    def is_valid(self):
        return self.data_present

    def get_wheel_data(self):
        # gets raw wheel data
        return self.sb.raw_mouse_wheel()

    def plot_wheel_data(self):
        # plots raw wheel data
        return self.sb.plot_raw_mouse_wheel()

    def plot_norm_data(self):

        # plot normalized wheel data
        data = self.normalize_wheel_data()

        plt.figure()
        plt.plot(data)
        plt.xlabel("Frames")
        plt.ylabel("norm wheel")
        fig1 = plt.gcf()
        return fig1

    def normalize_wheel_data(self):
        # since the fps of wheel data is about twice of the behavior video, we need to normalize
        # wheel data to the same fps

        # get wheel data
        wheel = self.get_wheel_data()
        wheel_indices = range(0, len(wheel))

        # get video frames
        frames = self.ep.get_per_frame_data()[0]

        # get video fps
        fps = self.sv.get_fps()

        fps_ratio= (len(wheel)/float(len(frames)))
        fps_wheel = fps*fps_ratio

        normal_wheel = []

        # initiate first frame
        normal_wheel.append(wheel[0])


        # For every behavior frame, get the closest wheel frame, the next and previous wheel frame,
        # and then add the average of these three values to the normalized wheel data
        for i in frames[1:len(frames)-1]:


            closest = wheel[int(i*fps_ratio)]
            next = wheel[int(i*fps_ratio + 1)]
            previous = wheel[int(i*fps_ratio -1)]
            avg = (closest + next + previous)/3.0
            normal_wheel.append(closest)

        # add the last wheel frame
        normal_wheel.append(wheel[-1])


        return normal_wheel