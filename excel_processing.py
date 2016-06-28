"""excel_processing.py by Mahdi Ramadan, 06-18-2016
This program will be used for excel file processing of
annotated behavior videos
"""
import os
import pandas
import sys

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

    def get_0_frames(self, label):
        data = self.get_column(label)
        return type(data)

