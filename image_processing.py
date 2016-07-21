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
import sys
import pickle
import time
from skimage.feature import hog
from skimage import data, color, exposure


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

    def run_whole_video(self):

        # wheel_data = self.wd.normalize_wheel_data()
        self.video_pointer.set(1, 36000)
        ret, frame = self.video_pointer.read()
        frames = []
        opticals = []
        angles = []
        prvs = cv2.cvtColor(frame[160:420, 100:640], cv2.COLOR_BGR2GRAY)
        ret, frame = self.video_pointer.read()
        # prvs = self.image_segmentation(frame)
        frames.append(prvs)
        hsv = np.zeros_like(frame[160:420, 100:640])
        hsv[..., 1] = 255
        count = 0

        while count < 5000:
            frame = cv2.cvtColor(frame[160:420, 100:640], cv2.COLOR_BGR2GRAY)
            next = frame
            # data = self.image_segmentation(frame)
            # frames.append(data['image'])
            optical = self.optical_flow(prvs, next)
            opticals.append(optical['mag'])
            angles.append(optical['ang'])
            hsv[..., 0] = optical['ang'] * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(optical['mag'], None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            cv2.imshow('frame2', rgb)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv2.imwrite('opticalfb.png', next)
                cv2.imwrite('opticalhsv.png', rgb)

            prvs = next
            ret, frame = self.video_pointer.read()
            count += 1

        self.video_pointer.release()
        cv2.destroyAllWindows()
            # prvs = data['image']



    def image_segmentation(self, frame):

        # tail = self.detect_tail(frame)

        # calculate frame dimensions
        height = len(frame)
        width = len(frame[1])

        # # select foreground
        # foreground = self.select_foreground(frame, width, height)

        # crop image
        where = 160
        crop = self.crop_image(frame, where)

        return {'image': crop}

    def show_frame(self, frame):
        cv2.imshow('image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def image_contrast(self, img_grey, alpha, beta):

        # increase contrast
        alpha = float(alpha)
        beta = float(beta)
        # alpha controls gain (contrast)
        self.array_alpha = np.array([alpha])
        # beta controls bias (brightness)
        self.array_beta = np.array([beta])

        # add a beta value to every pixel
        cv2.add(img_grey, self.array_beta, img_grey)

        # multiply every pixel value by alpha
        cv2.multiply(img_grey, self.array_alpha, img_grey)

        return img_grey

    def sharpen_image(self, image, value):

        kernel = np.zeros((9, 9), np.float32)
        # Identity, times two!
        kernel[4, 4] = value

        # Create a box filter:
        boxFilter = np.ones((9, 9), np.float32) / 81.0

        # Subtract the two:
        kernel = kernel - boxFilter

        # Note that we are subject to overflow and underflow here...but I believe that
        # filter2D clips top and bottom ranges on the output, plus you'd need a
        # very bright or very dark pixel surrounded by the opposite type.

        image = cv2.filter2D(image, -1, kernel)

        return image

    def select_foreground(self, frame, width, height):

        # convert to grey scale
        img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # apply gaussian blur filter
        gauss = cv2.GaussianBlur(img_grey, (7, 7), 0)

        # increase contrast
        img_contrast = self.image_contrast(gauss, 0.5, -100)

        # threshold
        ret, img = cv2.threshold(img_contrast, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # sharpen image
        img_sharp = self.sharpen_image(img, 8.0)

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(img_sharp, cv2.MORPH_OPEN, kernel, iterations=2)

        # select background
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # select foreground
        dist_transform = cv2.distanceTransform(opening, cv2.cv.CV_DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.05 * dist_transform.max(), 255, 0)

        # unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)


        return {'sure_fg': sure_fg, 'unknown': unknown}

    def crop_image(self, frame, where):

        # image processed without background (size is 480 x 640), based
        # on output of threshold function in select_foreground

        frame = frame[where:420, 100:640]

        return frame

    def optical_flow(self, prvs, next):

        # prvs = np.asarray(prvs)
        # next = np.asarray(next)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        return {'mag': mag, 'ang': ang}