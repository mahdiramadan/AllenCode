"""excel_process.py by Mahdi Ramadan, 06-27-2016
This program will be used to index and extrapolate data from excel files
containing behavior annotation data
"""

import pandas

class ExcelProcess:
    def __init__(self, excel_folder):
        # find excel folder
        for file in os.listdir(excel_folder):
            if file.endswith(".xlsx"):
                self.directory = exp_folder
                self.file_string = os.path.join(excel_folder, file)
                # data is of type DataFrame
                self.data = pandas.read_excel(self.file_string)
                self.data_present = os.path.isfile(self.file_string)

    def data_valid(self):
        return self.data_available

    def get_chattering (self):
        data_column = data['chattering'][1:]
        return data_column.size()
