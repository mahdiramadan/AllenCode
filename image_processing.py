"""image_processing.py by Mahdi Ramadan, 07-12-2016
This program will be used for video/image processing of
behavior video
"""
import os
import cv2



class ImageProcessing:
    def __init__(self, exp_folder, lims_ID):

        for file in os.listdir(exl_folder) and file.startswith(lims_ID):
            # looks for the excel file and makes the directory to it
            if file.endswith(".mp4"):
                self.directory = exp_folder
                self.file_string = os.path.join(exl_folder, file)

        if os.path.isfile(self.file_string):
                self.data_present = True
        else:
                self.data_present = False

    def is_valid(self):
        return self.data_present
