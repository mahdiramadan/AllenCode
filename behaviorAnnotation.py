"""behaviorAnnotation.py by Mahdi Ramadan, 06-18-2016
This program will be used for video annotation and display.
Referred to as behaviorAnnotation(1) by pycharm IDE
"""
# behaviorAnnotation.py must be in same folder as raw_behavior.py
from raw_behavior import RawBehavior as rb


raw = rb("/Users/mahdiramadan/Documents/Allen_Institute/code_repository/videos")

print(raw.get_xy_size())