"""behaviorAnnotation.py by Mahdi Ramadan, 06-18-2016
This program will be used for video annotation and display.
Referred to as behaviorAnnotation(1) by pycharm IDE
"""
# behaviorAnnotation.py must be in same folder as raw_behavior.py (to avoid adding files to path issues)

from raw_behavior import RawBehavior as rb
from stimulus_behavior import StimulusBehavior as sb
from synced_videos import SyncedVideos as sv
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import cv2

class DataAnalysis:
    def __init__(self,exp_folder):
        #
        for file in os.listdir(exp_folder):
            if file.endswith(".mp4"):
                # set up objects with parameters to be used
                self.directory = exp_folder
                self.file_string = os.path.join(exp_folder, file)
                self.data_pointer = cv2.VideoCapture(self.file_string)
                self.rb = rb(exp_folder)
                self.sb = sb(exp_folder)
                self.sv = sv(exp_folder)

            else:
                continue

        self.data_present = os.path.isfile(self.file_string)

    def data_valid(self):
        return self.data_available



# Actual running script

DataAnalysis = DataAnalysis("/Users/mahdiramadan/Documents/Allen_Institute/code_repository/Videos")
data = DataAnalysis.sb.raw_mouse_wheel()













