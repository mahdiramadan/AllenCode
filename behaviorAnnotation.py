"""behaviorAnnotation.py by Mahdi Ramadan, 06-18-2016
This program will be used for video annotation and display.
Referred to as behaviorAnnotation(1) by pycharm IDE
"""
# behaviorAnnotation.py must be in same folder as raw_behavior.py (to avoid adding files to path issues)

from raw_behavior import RawBehavior as rb
from stimulus_behavior import StimulusBehavior as sb
from synced_videos import SyncedVideos
import matplotlib.pyplot as plt
import numpy as np
import math

# class DataAnalysis:
#     def __init__(self):
#         #input acronym name of script wanted, and full method name
#         self.directory = "/Users/mahdiramadan/Documents/Allen_Institute/code_repository/Videos"
#         self.data_available = os.path.isfile(self.directory)
#         rb = rb(self.directory)
#         sb = sb(self.directory)
#         sv = sv(self.directory)
#
#     def data_valid(self):
#         return self.data_available
#
#     def execute_method(self,type,method):
#         return type + "."+ method +()

stim = SyncedVideos()
stim.video_annotation("/Users/mahdiramadan/Documents/Allen_Institute/code_repository/Videos")











