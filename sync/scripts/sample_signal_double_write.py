"""
Sample signal.  High speed pulse output for benchmarking.
"""
import time

import numpy as np

from toolbox.IO.nidaq import DigitalOutput



do = DigitalOutput("Dev1", port=1)
do.start()

high = np.array([1,1,1,1], dtype=np.uint8)
low = np.array([0,0,0,0], dtype=np.uint8)

do.write(high)
time.sleep(0.1)
do.write(low)
time.sleep(0.1)

do.stop()
do.clear()
