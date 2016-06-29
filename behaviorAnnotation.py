"""behaviorAnnotation.py by Mahdi Ramadan, 06-18-2016
This program will be used for video annotation and display.
Referred to as behaviorAnnotation by pycharm IDE
"""
# behaviorAnnotation.py must be in same folder as raw_behavior.py (to avoid adding files to path issues)

from raw_behavior import RawBehavior as rb
from stimulus_behavior import StimulusBehavior as sb
from synced_videos import SyncedVideos as sv
from excel_processing import ExcelProcessing as ep
from lims_database import LimsDatabase as ld
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import cv2
import sys
import pandas

class DataAnalysis:
    def __init__(self,exp_folder, lims_ID):
        #
                # set up objects with parameters to be used
                self.rb = rb(exp_folder)
                self.sb = sb(exp_folder)
                self.sv = sv(exp_folder)
                self.ep = ep(exp_folder)
                # self.ld = ld(lims_ID)

# Actual running script

# videos on this laptop stored in "/Users/mahdiramadan/Documents/Allen_Institute/code_repository/Videos"
# example LIMS ID is 501021421

# input LIMS ID or directory to files of interest!
# RawBehavior, Stimulusbehavior, SyncedVideos, ExcelProcessing take in video directory
# LimsDatabase takes in LIMS ID

# initializes all DataAnalysis objects, takes video directory and lims ID
DataAnalysis = DataAnalysis("/Users/mahdiramadan/Documents/Allen_Institute/code_repository/Videos", "501021421")

# data labels for annotation are: "ID", "From", "To", "chattering", "trunk_present", "grooming", "trunk_absent", "running"
# "startle", "tail_relaxed", "tail_tense", "flailing_present", "flailing_absent", "walking"

data = DataAnalysis.sv.video_annotation()













