"""excel_processing.py by Mahdi Ramadan, 06-18-2016
This program will be used for excel file processing of
annotated behavior videos
"""

import pandas

class DataAnalysis:
    def __init__(self,exl_folder):

        for file in os.listdir(exl_folder):
            if file.endswith(".xlsx"):
                self.directory = exl_folder
                self.file_string = os.path.join(exl_folder, file)
                # uploads as a DataFrame type
                self.data = pandas.read_excel(self.file_string)

            else:
                continue

        self.data_present = os.path.isfile(self.file_string)

    def data_valid(self):
        return self.data_available

    def get_chattering (self):
        chattering = data['chattering'][1:]
        return chattering