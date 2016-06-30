"""excel_processing.py by Mahdi Ramadan, 06-18-2016
This program will be used for excel file processing of
annotated behavior videos
"""
import os
import pandas
import sys
from raw_behavior import RawBehavior as rb
import cv2
import matplotlib.pyplot as plt
import numpy as np

class ExcelProcessing:
    def __init__(self, exl_folder):

        for file in os.listdir(exl_folder):
            if file.endswith(".xlsx") and not file.startswith("~$"):
                self.directory = exl_folder
                self.file_string = os.path.join(exl_folder, file)
                # uploads as a DataFrame type
                self.data = pandas.read_excel(self.file_string)

            else:
                continue

        self.data_present = os.path.isfile(self.file_string)
        self.behavior_data_flow = rb

    def data_valid(self):
        return self.data_available

    def get_id(self):
        ID = self.data['ID']
        return ID

    # IMPORTANT: indexing starts with first row of data at zero. Thus, the ID number and index
    # number of a data point are offset by one


    def get_column(self, label):
        # method to extract column of data based on label
        data = self.data[label]
        return data

    def get_categories(self):
        # returns data labels
        categories = list(self.data.columns.values)
        return categories

    def get_size(self):
        # returns size of data (ignoring labels in first row)
        ID_length = len(self.get_id())
        column_length = len(self.get_categories())
        return (ID_length, column_length)

    def get_from(self):
        # returns "from" frames data
        f = self.data['From']
        return f

    def get_to(self):
        # returns "to" frames data
        f = self.data['To']
        return f

    def get_true_false(self, label, index):
        data = self.get_column(label)
        if data[index] == 0:
            return 0
        if data[index] == 1:
            return 1

    def get_0_frames(self, label):
        # returns all the frames that the label specified was equal to zero
        data = self.get_column(label)
        count = 0
        frames = [[], []]
        for i in data:
            if i == 0:
                frames[0].append(self.get_from()[count])
                frames[1].append(self.get_to()[count])
                count += 1

            else:
                count += 1
                continue

        return frames

    def get_1_frames(self, label):
        # returns all the frames that the label specified was equal to one
        data = self.get_column(label)
        count = 0
        frames = [[], []]
        for i in data:
            if i == 1:
                frames[0].append(self.get_from()[count])
                frames[1].append(self.get_to()[count])
                count += 1

            else:
                count += 1
                continue

        return frames

    def get_frame_start(self,count):
        data = self.get_column("From")[count]
        return data

    def get_frame_end(self,count):
        data = self.get_column("To")[count]
        return data

    def video_annotation_labels(self):

        # outputs a .mp4 video with frame number and labeled annotation text
        file_name = rb("/Users/mahdiramadan/Documents/Allen_Institute/code_repository/Videos").get_file_string()
        data_pointer = cv2.VideoCapture(file_name)
        fps = data_pointer.get(cv2.cv.CV_CAP_PROP_FPS)
        nFrames = int(data_pointer.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        frameWidth = int(data_pointer.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        frameHeight = int(data_pointer.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.cv.CV_FOURCC(*'mp4v')
        # change 3rd parameter of out function for different playback speeds
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (frameWidth, frameHeight))
        ret, frame = data_pointer.read()

        # gets the data table with frame number and 0 or 1 for each column label
        frame_data = self.get_per_frame_data()
        # iterates through each row
        for i in range(self.get_first_frame(), self.get_last_frame()):
            # prints frame number
            cv2.putText(img=frame,
                        text=str(int(data_pointer.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))),
                        org=(20, 100),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1,
                        color=(0, 255, 0),
                        thickness=2,
                        lineType=cv2.CV_AA)
            count2 = 0
            # iterates for each column label
            for k in range(len(self.get_labels())):
                # if true (value = 1), then we print label
                if frame_data[k+1][i] == 1:
                    count2 += 1
                    # prints text in green or red depedning on column label
                    if k == 0 or k == 3 or k == 6:
                        c = (0,0,255)
                    else:
                        c = (0,255,0)

                    cv2.putText(img=frame,
                                text=str(self.get_labels()[k]),
                                org=(0+count2*100, 100),
                                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                                fontScale=0.5,
                                color= c,
                                thickness=1,
                                lineType=cv2.CV_AA)
                else:
                    continue

            # write out the frame
            out.write(frame)
            # read next frame
            ret, frame = data_pointer.read()

        # if number of labeled frames is less than number of video frames, just print frame number
        while ret:
            cv2.putText(img=frame,
                        text=str(int(data_pointer.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))),
                        org=(20, 100),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1,
                        color=(0, 255, 0),
                        thickness=2,
                        lineType=cv2.CV_AA)
            out.write(frame)
            # read next frame
            ret, frame = data_pointer.read()



    def get_per_frame_data(self):
        # initiates data lists
        frame_start = []
        frame_end = []
        labels = self.get_labels()
        # initiates the labels x number of frames data list
        frame_data = [[] for _ in range(len(self.get_labels()) + 1)]
        # first column set to frame numbers between first and last frame
        frame_data[0].extend(range(self.get_first_frame(), self.get_last_frame()))

        # initiates all column labels to either 0 or 1
        for k in range(len(self.get_labels())):
            frame_data[k + 1].append(self.get_true_false(labels[k], 0))

        # gets the frame start and end of each row in the excel file
        for p in range(len(self.get_column("From"))):
            frame_start.insert(0, self.get_frame_start(p))
            frame_end.insert(0, self.get_frame_end(p))

            # for each frame, puts a 0 or 1 for each column label
            for k in range(len(self.get_labels())):
                frame_data[k + 1].extend([self.get_true_false(self.get_labels()[k], p)] * (frame_end[0] - frame_start[0]))

        return frame_data

    def get_first_frame(self):
        # first labeled frame
        first = self.get_column("From").iget(0)
        return first

    def get_last_frame(self):
        # last labeled frame
        last = self.get_column("To").iget(-1) + 1
        return last

    def get_labels(self):
        # column labels
        labels = ["chattering", "down", "grooming", "moving", "relaxed", "running", "startle", "tailrelaxed",
                  "tailtense", "tense", "up", "walking"]
        return labels

    def get_name(self):
        # returns name of annotator
        name = self.get_column("name").iget(0)
        return name

    def get_mouse_ID(self):
        # returns Mouse LIMS ID
        name = self.get_column("name").iget(0)
        return name

    def get_date(self):
        # returns date of annotation
        date= self.get_column("name").iget(0)
        return date

    def get_bar_plot(self,label):
        # returns bar plot of frame number and occurrence (0 vs. 1) of a specified column label
        data = self.get_per_frame_data()
        frames = data[0]
        label_data = data[self.get_labels().index(label)]

        fig1 = plt.figure()
        fig1.suptitle('Occurrence vs. Frame Number', fontsize=14, fontweight='bold')
        ax = fig1.add_subplot(111)
        ax.set_xlabel('frame number')
        ax.set_ylabel('Occurrence')
        ax.bar(frames, label_data)

        return fig1

    def get_cumulative_plot(self, label):
        # returns cumulative sum plot of label occurrence vs. frame number
        data = self.get_per_frame_data()
        frames = data[0]
        label_data = np.cumsum(data[self.get_labels().index(label)])

        fig1 = plt.figure()
        fig1.suptitle('CumSum of Occurrence vs. Frame Number', fontsize=14, fontweight='bold')
        ax = fig1.add_subplot(111)
        ax.set_xlabel('frame number')
        ax.set_ylabel('CumSum of Occurrence')
        ax.bar(frames, label_data)

        return fig1

