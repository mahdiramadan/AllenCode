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
import sys
import time
import scipy.optimize as optimization



class ImageProcessing:
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

    def select_foreground(self,frame, width, height):

        # convert to grey scale
        img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # apply gaussian blur filter
        gauss = cv2.GaussianBlur(img_grey, (7, 7), 0)

        # increase contrast
        img_contrast = self.image_contrast(gauss, 0.5, -100)

        # threshold
        ret, img = cv2.threshold(img_contrast, 200, 255, cv2.THRESH_OTSU)

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

        return sure_fg


    def image_segmentation(self):

        # get frame at specified frame number
        self.video_pointer.set(1, 62000)
        ret, frame = self.video_pointer.read()


        # calculate frame dimensions
        height= len(frame)
        width= len(frame[1])

        # select foreground
        foreground = self.select_foreground(frame, width, height)

        # crop image
        where = 140
        crop = self. crop_image(frame, foreground, width, height, where)


        # select mouse
        mouse = self.select_image_range(crop, crop, width, height)

        # if the cropped out image is too small (due to too much cropping), re-crop using a different
        # where value. If nothing works, print error message
        while mouse['range'] < 180:
            try:
                where = where - 10
                crop = self.crop_image(frame, foreground, width, height, where)
                mouse = self.select_image_range(crop, crop, width, height)

            except:
                print ('mouse height position is unexpected')
                sys.exit()

        # find center of mouse
        center = self.find_mouse_center(mouse['image'])

        # crop out wheel
        no_wheel = self.crop_wheel(mouse['image'], mouse['rawimage'], mouse['xpoints'], mouse['ypoints'], mouse['width'], mouse['height'])

        tail = detect_tail(frame)

        # fill mouse up from reference point generate by find_mouse_center
        mouse_image = self.fill_mouse(mouse['image'], no_wheel, center['x_center'], center['y_center'], mouse['initial'])



    def crop_image(self, frame, sure_fg, width, height, where):

        # image processed without background (size is 480 x 640), based
        # on output of threshold function in select_foreground
        for i in range(0, height):
            # where determines how much top of image is removed
            if i in range(0, where):
                frame[i, :] = 255
            else:
                for j in range(0, width):
                    if sure_fg[i, j] != 0:
                        frame[i, j] = 255
        return frame

    def select_image_range(self, frame, image, w, h):

        # edge detection algorithm
        image = cv2.Canny(image, 60, 100)

        # initial frame to use as reference (should be dark)
        initial = image[0, w-1]
        top = 0
        # starting from the top of the image, iterate down until
        # pixel value changes over 50% from initial pixel.
        # this pixel is now the new top of the image
        for height in range(0, h):

            if image[height, w-1] - initial > initial - initial/2:
                top = height

                break

        bottom = 0

        # starting at the bottom, iterate up the image until pixel value changes
        # over 50% from initial pixel. This is now the new bottom of image
        for up in range(h-1, 0, -1):
            if image[up, w/2: w-1].max() - initial > initial - initial / 2:
                bottom = up
                break

        # create new image based on new top and bottom
        raw_image = frame[top+10:bottom, 0:w]
        new_image = image[top + 10:bottom, 0:w]

        height = len(new_image)
        width = len(new_image[1])

        points = []

        x = 0
        y = height-1

        # Wheel edge detection method. Starting at the bottom of the image for width values spaced 63 pixels apart,
        # iterate up the image. When the mean pixel value of the 63 wide sweep is over 50% different than
        # the initial frame, label point as part of edge

        while (x <= width):
            if new_image[y, x:(x+63)].mean() - initial > initial - initial / 2:
                points.append([x+63, y])
                x = x + 63
                y = height-1

            else:
                if y != 0:
                    y = y-1
                else:
                    x= x + 63
                    y = height-1

        xs = []
        ys= []

        # return x and y coordinates for points outlining wheel edge
        for point in points:
            xs.append(point[0])
            ys. append(point[1])
            # cv2.circle(new_image, (point[0], point[1]), 10, 255)

        # self.show_frame(new_image)


        return {'image':new_image,'rawimage': raw_image, 'range':bottom-top, 'xpoints': xs, 'ypoints':ys, 'width': width, 'height': height, 'initial': initial}


    def find_mouse_center(self, image):
        # this method will find a point on the mouse (close to center of mouse) based on the center of
        # mass of image edges
        xs = []
        ys = []

        height = len(image)
        width = len(image[1])

        # get points where edges exist (colored white)
        for i in range(0, width):
            for k in range(0, height):
                if image[k, i] == 255:
                    xs. append(i)
                    ys.append(k)

        # calculate center of gravity for x and y
        x_center = int(np.sum(xs)/float(len(xs)))
        y_center = int(np.sum(ys)/float(len(ys)))

        cv2.circle(image, (x_center, y_center), 10, 255)
        self.show_frame(image)



        return {'x_center': x_center, 'y_center': y_center}

    def fill_mouse(self, image, raw_image, x_center, y_center, initial):

        if






        xl= x_center
        xr = x_center
        y = np.array(range(y_center-5, y_center+5))
        count = 0
        mod = 4
        l = 0
        while xr - xl < 250 and l < 10:
            l += 1
            while not image[y, xr].mean() - initial > initial / 2:
                xr= xr+ 1
                count += 1
                if count%mod == 0:
                    y = y + 1

            y2 = np.array(range(y_center - 5, y_center + 5))
            while not image[y2, xl].mean() - initial > initial / 2:
                xl = xl- 1
                count += 1

            if xr-xl < 200:
                mod = mod + 2
            if xr-xl > 400:
                print ('did not find mouse back edge')
                break

        # cv2.circle(image, (xl, y2[0]), 10, 255)
        # cv2.circle(image, (xr, y[0]), 10, 255)

        y = y_center
        xs= []
        ys= []
        count = 0
        interval = 20
        for x in range(xl,xr, interval):
            while not image[y, x+1:x+interval].mean() - initial > initial / 2 and count < y_center:
                count += 1
                y = y - 1

            count = 0
            xs.append(x+interval)
            ys.append(y)
            y_center  = y_center + (interval/mod)/2
            y = y_center

        for item in range(len(xs)):
            cv2.circle(image, (xs[item], ys[item]), 10, 255)
        self.show_frame(image)

        x = xs[0] - interval
        y = ys[0]

        while not image[y, x].max() - initial > initial / 2:
            x = x -1

        y_init= y
        count = 0
        while y-y_init < 20:
            count += 1
            while image[y, x].mean() - initial > initial / 2:
                y = y + 1

            while not image[y, x].max() - initial > initial / 2:
                y= y + 1

            if count > 100:
                print ('Could not find mouth')
                break

        print(x,y)

        # cv2.circle(image, (x, y), 10, 255)
        self.show_frame(image)

    def detect_tail(self, frame):

        self.show_frame(frame)
        


    def crop_wheel(self, image, rawimage, xpoints, ypoints, width, height):

        # this method will crop the wheel out based on the edge detection algorithm
        # employed in select_image_range

        # initialize residual value
        residual = width*height

        # while the residual value is over a threshold, change which points are used to construct wheel edge
        while residual > 2000:
            # format x points into matrix
            A = np.vstack([xpoints, np.ones(len(xpoints))]).T
            # calculate a least square fit line to x and y points on wheel edge
            m, c = np.linalg.lstsq(A, ypoints)[0]
            # return distance residual of points
            residual = np.linalg.lstsq(A, ypoints)[1]

            # if there is too much residual (due to tail bent onto wheel or noise on wheel), find which
            # points are most responsible for the residual
            if residual > 2000:
                distance = 0
                max = 0
                xmax = 0
                ymax = 0
                index = 0
                for i in range(len(xpoints)):
                    distance = np.absolute(m*xpoints[i] + c - (ypoints[i]))
                    if distance > max:
                        max = distance
                        xmax = xpoints[i]
                        ymax = ypoints[i]
                        index = i
                # if the points responsible for the residual are in the first half of the image (left to right),
                # delete these points (most likely not the tail)
                if xmax < 320:
                    del xpoints[index]
                    del ypoints[index]
                # if the points are in the second half of the image, change slope of line to accommodate (probably
                # is the tail)
                elif xmax >= 320:
                    # change slope to 0.3
                    m = 0.30
                    break
        # add ten to intersect to allow for some room from edge
        c = c + 10

        # visualize line constructed
        first_point = int(m*xpoints[0] + c)
        last_point = np.absolute(int(m*xpoints[-1] + c))
        cv2.line(image, (0,first_point), (width,last_point), (255,255,255))
        # self.show_frame(image)

        # crop out wheel by checking if pixels are below or above line constructed
        # at edge of wheel
        for i in range (int(c), height-1):
            for k in range (0, width):
                if i > m*k + c:
                    rawimage[i,k] = 255

        self.show_frame(rawimage)
        return rawimage


def func(x, a, b, c, d, e,f,g):
    return a*x*x*x*x*x*x + b*x*x*x*x*x + c*x*x*x*x + d*x*x*x + e*x*x + f*x + g











