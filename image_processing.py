"""image_processing.py by Mahdi Ramadan, 07-12-2016
This program will be used for video/image processing of
behavior video
"""
import os
from stimulus_behavior import StimulusBehavior as sb
from excel_processing import ExcelProcessing as ep
from synced_videos import SyncedVideos as sv
from wheel_data import WheelData as wd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas
import time
import tables
import h5py



class ImageProcessing:
    def __init__(self, exp_folder, lims_ID):

        for file in os.listdir(exp_folder):
            # looks for the excel file and makes the directory to it
            if file.endswith(".mp4") and file.startswith(lims_ID):
                self.directory = exp_folder
                self.file_string = os.path.join(exp_folder, file)
                self.sb = sb(exp_folder)
                self.wd = wd(exp_folder, lims_ID)
                self.ep = ep(exp_folder, lims_ID)
                self.sv = sv(exp_folder, lims_ID)
                self.video_pointer = cv2.VideoCapture(self.file_string)

        if os.path.isfile(self.file_string):
            self.data_present = True
        else:
            self.data_present = False

    def is_valid(self):
        return self.data_present

    def show_frame(self, frame):
        cv2.imshow('image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run_whole_video(self):

        # wheel_data = self.wd.normalize_wheel_data()

        hf = h5py.File('data.h5', 'w')

        # self.video_pointer.set(1, 30000)
        ret, frame = self.video_pointer.read()

        frame = cv2.cvtColor(frame[160:420, 100:640], cv2.COLOR_BGR2GRAY)

        prvs = frame
        next = frame

        count = 0
        mod = 0
        opticals = []
        angles = []
        frames = []

        limit = self.wd.wheel_data_length()

        while count <= limit:
            if count > 0:
                if count%1000 == 0 or count == limit:
                    mod += 1
                    g = hf.create_group('first ' + str(mod) + '000 frames')
                    g.create_dataset('frames', data=frames, compression="gzip", compression_opts=9)
                    g.create_dataset('optical', data=opticals, compression="gzip", compression_opts=9)
                    g.create_dataset('angles', data=angles, compression="gzip", compression_opts=9)


                    frames = []
                    opticals = []
                    angles = []

            prvs = next
            frames.append(prvs)

            ret, frame = self.video_pointer.read()
            next = cv2.cvtColor(frame[160:420, 100:640], cv2.COLOR_BGR2GRAY)

            optical = self.optical_flow(prvs, next)
            opticals.append(optical['mag'])
            angles.append(optical['ang'])

            count += 1

            if count%1000 == 0:
                print (count)

    def optical_flow(self, prvs, next):


        flow = cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang= cv2.cartToPolar(flow[..., 0], flow[..., 1])

        mag = np.int16(mag)
        ang = np.int16(ang*180/np.pi/2)

        return {'mag': mag, 'ang': ang}