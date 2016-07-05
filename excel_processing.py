"""excel_processing.py by Mahdi Ramadan, 06-18-2016
This program will be used for excel file processing of
annotated behavior videos
"""
import os
import pandas
import sys
from raw_behavior import RawBehavior as rb
from synced_videos import SyncedVideos as sv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm

class ExcelProcessing:
    def __init__(self, exl_folder, lims_ID):

        for file in os.listdir(exl_folder):
            # looks for the excel file and makes the directory to it
            if file.endswith(".xlsx") and not file.startswith("~$"):
                self.directory = exl_folder
                self.file_string = os.path.join(exl_folder, file)
                self.lims_ID = lims_ID
                # excel uploads as a DataFrame type
                # IMPORTANT: zero based excel indexing starts with first row of numbered DATA, not the column labels.
                # This means row zero is the first row of actual Data, not the column labels
                self.data = pandas.read_excel(self.file_string)
                # get video directory and name
                file_name = rb(exl_folder,lims_ID).get_file_string()
                # data pointer to the behavior video annotated
                self.data_pointer = cv2.VideoCapture(file_name)

            else:
                continue

        self.data_present = os.path.isfile(self.file_string)
        self.behavior_data_flow = rb

    def data_valid(self):
        return self.data_available

    def frames_continuous(self):
        # this method checks to see if the labeled frames are continuous. Make sure the labeled frame data is continuous
        # for the rest of the code to work!
        # gets the to and from frames
        To_frames = self.get_to()
        From_frames = self.get_from()

        # for the each iteration, check whether the "to" frame is equal to the "from" frame in the next row
        # if not continuous, returns which rows are discontinuous
        for k in range(len(From_frames)-1):
            if To_frames[k] != From_frames[k+1]:
                return "Frames are not continuous between row number %r and %r of the data" % (k+2, k+3)
            else:
                continue
        return "Frames are continuous!"



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

    def get_zero_frames_range(self, label):
        # returns all the frame ranges that the label specified was equal to zero
        data = self.get_column(label)
        count = 0
        frames = [[], []]
        # Note: first column is all the From frames, second is all the To frames
        for i in data:
            if i == 0:
                frames[0].append(self.get_from()[count])
                frames[1].append(self.get_to()[count])
                count += 1

            else:
                count += 1
                continue

        return frames

    def get_one_frames_range(self, label):
        # returns all the frame ranges that the label specified was equal to one
        data = self.get_column(label)
        count = 0
        frames = [[], []]
        # NOTE: first column is all the From frames, second is all the To frames
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

        # gets video file information
        fps = self.data_pointer.get(cv2.cv.CV_CAP_PROP_FPS)
        nFrames = int(self.data_pointer.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        frameWidth = int(self.data_pointer.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        frameHeight = int(self.data_pointer.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.cv.CV_FOURCC(*'mp4v')
        # change 3rd parameter of out function for different playback speeds
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (frameWidth, frameHeight))
        ret, frame = self.data_pointer.read()

        # gets the data table with frame number and 0 or 1 for each column label
        frame_data = self.get_per_frame_data()
        # iterates through each row
        for i in range(self.get_first_frame(), self.get_last_frame()+1):
            # prints frame number on frame
            cv2.putText(img=frame,
                        text=str(int(self.data_pointer.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))),
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
                    # prints text in green or red depending on column label
                    if k == 0 or k == 3 or k == 6:
                        c = (0,0,255)
                    else:
                        c = (0,255,0)

                    cv2.putText(img=frame,
                                text=str(self.get_labels()[k]),
                                org=(0+count2*120, 100),
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
            ret, frame = self.data_pointer.read()

        # if number of labeled frames is less than number of video frames, just print frame number
        while ret:
            cv2.putText(img=frame,
                        text=str(int(self.data_pointer.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))),
                        org=(20, 100),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1,
                        color=(0, 255, 0),
                        thickness=2,
                        lineType=cv2.CV_AA)
            out.write(frame)
            # read next frame
            ret, frame = self.data_pointer.read()



    def get_per_frame_data(self):
        # This method takes in the annotated excel data with frame ranges, and returns a data matrix of
        # each frame number annotated, along with the annotation scheme of each label (0 vs. 1) at that frame

        # initiates data lists
        frame_start = []
        frame_end = []
        labels = self.get_labels()
        # initiates the labels x number of frames data list, the minus 3 is to ignore the columns of
        # name, mousid and date
        frame_data = [[] for _ in range(len(self.get_labels()) + 1)]
        # first column set to frame numbers between first and last frame
        frame_data[0].extend(range(self.get_first_frame(), self.get_last_frame()+1))

        # initiates all column labels to either 0 or 1
        for k in range(len(self.get_labels())):
            frame_data[k + 1].append(self.get_true_false(labels[k], 0))

        # gets the frame start and end of each row in the excel file
        for p in range(len(self.get_column("From"))):
            frame_start.insert(0, self.get_frame_start(p))
            frame_end.insert(0, self.get_frame_end(p))

            # for each frame, puts a 0 or 1 for each column label
            # if you have frames 0 to 10 == 1, 10 to 20 == 0, frame 10 == 1 due to how code is set-up
            for k in range(len(self.get_labels())):
                frame_data[k + 1].extend([self.get_true_false(self.get_labels()[k], p)] * (frame_end[0] - frame_start[0]))

        return frame_data

    def get_first_frame(self):
        # first labeled frame
        first = self.get_column("From").iget(0)
        return first

    def get_last_frame(self):
        # last labeled frame
        last = self.get_column("To").iget(-1)
        return last

    def get_labels(self):
        # column labels of interest for data (ignoring name, lims id, date)
        # make sure to update if columns change
        labels = ["chattering", "trunk_present", "grooming", "trunk_absent", "running",
                  "startle", "tail_relaxed", "tail_tense", "flailing_present", "flailing_absent", "walking"]
        return labels

    def get_name(self):
        # returns name of annotator
        name = self.get_column("person").iget(0)
        return name

    def get_mouse_ID(self):
        # returns Mouse LIMS ID
        name = self.get_column("mouseid").iget(0)
        return name

    def get_date(self):
        # returns date of annotation
        date= self.get_column("timestamp").iget(0)
        return date

    def get_bar_plot(self,label):
        # returns bar plot of frame number and occurrence (0 vs. 1) of a specified column label
        data = self.get_per_frame_data()
        # get frame data
        frames = data[0]
        # get column data
        label_data = data[self.get_labels().index(label)]

        # create plot and axis
        fig1 = plt.figure()
        fig1.suptitle('Occurrence vs. Frame Number', fontsize=14, fontweight='bold')
        ax = fig1.add_subplot(111)
        ax.set_xlabel('frame number')
        ax.set_ylabel('Occurrence')
        ax.bar(frames, label_data)

        return fig1

    def get_cumulative_plot_frame(self, label):
        # returns cumulative sum plot of label occurrence vs. frame number
        data = self.get_per_frame_data()
        frames = data[0]
        # get cumsum of column data
        label_data = np.cumsum(data[self.get_labels().index(label) + 1])
        # create plot axis and figure
        fig1 = plt.figure()
        fig1.suptitle('CumSum of Occurrence vs. Frame Number', fontsize=14, fontweight='bold')
        ax = fig1.add_subplot(111)
        ax.set_xlabel('frame number')
        ax.set_ylabel('CumSum of Occurrence')
        ax.bar(frames, label_data)

        return fig1

    def get_frequency_plot(self, label):
        # returns a plot of the change in frequency of a label per SECOND
        data = self.get_per_frame_data()
        fps = sv(self.directory, self.lims_ID).get_fps()
        # get cumsum of column data
        label_data = np.cumsum(data[self.get_labels().index(label) + 1])
        # initiate counters and lists
        frequency_data = []
        time = []
        count = 0
        n = 0
        # determines over how many frames we calculate annotation frequency
        interval = 147

        # iterate over each frame
        for k in range(len(label_data)):
            count += 1
            # if count mod interval size = equal, then calculate the difference of the cumsum associated
            # with this frame minus the cumsum at the frame one interval before
            if count % interval == 0:
                frequency_data.append(label_data[k]-label_data[k-(interval-1)])
                n += 1
                # round time in seconds to nearest integer
                second = round(n*interval/fps)
                time.append(second)
            else:
                continue

        # create plot axis and figure

        fig1 = plt.figure()
        fig1.suptitle('Frequency of Occurrence, sampled every 5 seconds', fontsize=14, fontweight='bold')
        ax = fig1.add_subplot(111)
        ax.set_xlabel('Time (Sec)')
        ax.set_ylabel('Frequency of Occurrence')
        ax.bar(time, frequency_data)

        return fig1

    def store_frame_data(self, label):
        # stores frame data according to whether specified label has 0 or 1 on that frame
        frame_pictures = [[] for _ in range(2)]
        frame_data = self.get_per_frame_data()


        # Iterate through every frame
        for i in range(len(frame_data[0])):
            # i +1 because first frame is at position 1, not 0 in OpenCV
            # read frame at ith position
            self.data_pointer.set(1,i+1)
            ret, frame = self.data_pointer.read()
            # if the label associated with the current frame is equal to zero, place in first column of
            # frame_data, otherwise if its equal to 1 place in second column of frame_data
            # +1 because 0th column on frame_data is frame numbers, label column start at column 1
            if frame_data[self.get_labels().index(label) + 1][i] == 0:
                frame_pictures[0].append(frame)
            elif frame_data[self.get_labels().index(label) +1][i] == 1:
                frame_pictures[1].append(frame)
            else:
                continue

        return frame_pictures

    def is_from_smaller_than_to(self):

        # makes sure the from frame is smaller than the to frame number in a row
        To_frames = self.get_to()
        From_frames = self.get_from()
        # iterates through all rows and does check
        for k in range(len(From_frames)):
            if From_frames[k] == To_frames[k] or From_frames[k] > To_frames[k]:
                return "Frames are timed incorrectly in row %r of the data" %(k+2)
            else:
                continue
        return "All frames are timed correctly!"