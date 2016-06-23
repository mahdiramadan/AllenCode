"""behaviorAnnotation.py by Mahdi Ramadan, 06-18-2016
This program will be used for video annotation and display
"""

from code_repo.raw_behavior import RawBehavior as rb


raw = rb("/Users/mahdiramadan/Documents/Allen_Institute/code_repository/videos", "/Users/mahdiramadan/Documents/Allen_Institute/code_repository/Videos/temp_folder")

raw.get_image()