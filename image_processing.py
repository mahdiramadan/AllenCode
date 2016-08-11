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
from sklearn import preprocessing
from multiprocessing import Pool
import warnings



def show_frame(frame):
    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_whole_video(exp_folder, lims_ID):

    file_string = get_file_string(exp_folder, lims_ID)
    video_pointer = cv2.VideoCapture(file_string)

    hf = h5py.File('data_'+str(lims_ID)+'.h5', 'w')

    # self.video_pointer.set(1, 41000)
    ret, frame = video_pointer.read()


    frame = cv2.cvtColor(frame[160:400, 100:640], cv2.COLOR_BGR2GRAY)

    prvs = frame
    next = frame

    count = 0
    mod = 0
    opticals = []
    angles = []
    frames = []


    limit = int(video_pointer.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

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
        frames.append(process_input(prvs))

        ret, frame = video_pointer.read()
        next = cv2.cvtColor(frame[160:400, 100:640], cv2.COLOR_BGR2GRAY)

        optical = optical_flow(prvs, next)
        opticals.append(optical['mag'])
        angles.append(optical['ang'])

        count += 1

        if count%1000 == 0:
            print (count)

def optical_flow(prvs, next):


    flow = cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang= cv2.cartToPolar(flow[..., 0], flow[..., 1])

    mag = process_input(mag)
    ang = process_input((ang*180/np.pi/2))

    return {'mag': mag, 'ang': ang}

def process_input(input):

    frame_data = []

    for (x, y, window) in sliding_window(input, 30, (30, 30)):
        hist, bin = np.histogram(window, 10)
        center = (bin[:-1] + bin[1:]) / 2
        hist_x = np.multiply(center, hist)
        hist_x = preprocessing.MinMaxScaler((-1, 1)).fit(hist_x).transform(hist_x)
        frame_data = np.concatenate((frame_data, hist_x))
    return frame_data

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def get_file_string(exp_folder,lims_ID):

    for file in os.listdir(exp_folder):
        if file.endswith(".mp4") and file.startswith(lims_ID):
            file_string = os.path.join(exp_folder, file)
            return file_string



if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    exp_folder = '/Users/mahdiramadan/Documents/Allen_Institute/code_repository/Data/501560436'
    lims_ID = '501560436'
    run_whole_video(exp_folder, lims_ID)

    # p = Process(target=run_whole_video(video_pointer, lims_ID), args= (video_pointer, lims_ID))
    # p.start()
    # p.join()

