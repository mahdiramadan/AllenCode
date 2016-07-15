"""image_processing.py by Mahdi Ramadan, 07-12-2016
This program will be used for video/image processing of
behavior video
"""
import os
from stimulus_behavior import StimulusBehavior as sb
from excel_processing import ExcelProcessing as ep
from synced_videos import SyncedVideos as sv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage.feature import greycomatrix, greycoprops
import tqdm
import time
from skimage import data
import scipy
from scipy.optimize import minimize
from scipy.spatial import distance
import math


class ImageProcessing:
    def __init__(self, exp_folder, lims_ID):

        for file in os.listdir(exp_folder):
            # looks for the excel file and makes the directory to it
            if file.endswith(".mp4") and file.startswith('output'):
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

    def select_foreground(self,frame):

        # convert to grey scale
        img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # apply gaussian blur filter
        gauss = cv2.GaussianBlur(img_grey, (7, 7), 0)

        # increase contrast
        img_contrast = self.image_contrast(gauss, 0.5, -100)

        # threshold
        ret, img = cv2.threshold(img_contrast, 220, 255, cv2.THRESH_OTSU)

        # sharpen image
        img_sharp = self.sharpen_image(img, 8.0)

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(img_sharp, cv2.MORPH_OPEN, kernel, iterations=2)

        # background
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # foreground
        dist_transform = cv2.distanceTransform(opening, cv2.cv.CV_DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.05 * dist_transform.max(), 255, 0)

        # unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # image processed without background (size is 480 x 640)
        for i in range(0, 480):
            if i in range(0, 140) or i in range(400, 480):
                img_grey[i, :] = 255
            else:
                for j in range(0, 640):
                    if sure_fg[i, j] != 0:
                        img_grey[i, j] = 255

        return img_grey


    def image_segmentation(self):

        # get frame
        ret, frame = self.video_pointer.read()

        # select foreground
        foreground = self.select_foreground(frame)
        # select mouse
        mouse = self.select_mouse(foreground)


    def select_mouse(self):
        ret, frame = self.video_pointer.read()

        self.gradient_descent(frame)

        # dist = 0
        # alpha = 0
        # beta = 0
        # zet = 0
        # for m in range(-10,10):
        #     for k in range (-100, 100):
        #         for l in range (0, 20):
        #             v = self.generate_glcm(m,k,l, frame)
        #             if v > dist:
        #                 dist = v
        #                 alpha = m
        #                 beta = k
        #                 zet = l
        # print(dist, alpha, beta, zet)


    def generate_glcm(self, x, y, z, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        PATCH_SIZE = 150

        # increase contrast
        alpha = float(x)
        beta = float(y)
        # alpha controls gain (contrast)
        self.array_alpha = np.array([alpha])
        # beta controls bias (brightness)
        self.array_beta = np.array([beta])

        # add a beta value to every pixel
        cv2.add(img, self.array_beta, img)

        # multiply every pixel value by alpha
        cv2.multiply(img, self.array_alpha, img)

        kernel = np.zeros((9, 9), np.float32)
        # Identity, times two!
        kernel[4, 4] = z

        # Create a box filter:
        boxFilter = np.ones((9, 9), np.float32) / 81.0

        # Subtract the two:
        kernel = kernel - boxFilter

        # Note that we are subject to overflow and underflow here...but I believe that
        # filter2D clips top and bottom ranges on the output, plus you'd need a
        # very bright or very dark pixel surrounded by the opposite type.

        image = cv2.filter2D(img, -1, kernel)

        img = image

        head_locations = [(150, 200), (150, 250), (150, 400), (150, 600), (150, 560), (150, 500), (160, 460),
                          (170, 420)]
        head_patches = []
        for loc in head_locations:
            head_patches.append(img[loc[0]:(loc[0] + PATCH_SIZE),
                                loc[1]:(loc[1] + PATCH_SIZE)])
            # To visualize areas, uncomment line below
            # cv2.circle(image, (loc[1], loc[0]), 10, 255)

        mouse_locations = [(200, 200), (200, 300), (200, 400), (300, 400), (300, 500), (320, 550), (320, 480),
                           (230, 200)]
        mouse_patches = []
        for loc in mouse_locations:
            mouse_patches.append(img[loc[0]:(loc[0] + PATCH_SIZE),
                                 loc[1]:(loc[1] + PATCH_SIZE)])
            # cv2.circle(image2, (loc[1], loc[0]), 10, 255)

        xs = []
        ys = []

        for patch in (head_patches + mouse_patches):
            glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
            xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
            ys.append(greycoprops(glcm, 'correlation')[0, 0])

        head = []
        mouse = []

        for i in range(len(head_patches)):
            head.append((xs[i], ys[i]))

        for i in range(len(head_patches), len(xs)):
            mouse.append((xs[i], ys[i]))

        dist = []
        for i in range(len(head)):
                k = head[i]
                l = mouse[i]
                dist.append(math.sqrt(math.pow((k[0]-l[0]), 2) + math.pow((k[1] - l[1]), 2)))
        total_dist = sum(dist)

        return total_dist


    def gradient_descent(self, frame):

        x = -10
        y = -100
        z = 0
        limit = 1000

        dist_old = self.generate_glcm(x,y,z,frame)
        dist_new = 100
        step = 10
        count = 0

        while (count) < limit:
            dist_new = self.generate_glcm(x + step, y + step, z + step, frame)
            if dist_new >= dist_old:
                x = x + step
                y = y + step
                z = z + step
                dist_old = dist_new

            else:
                step = step-1

            count += 1
        print (x,y,z, dist_new)











