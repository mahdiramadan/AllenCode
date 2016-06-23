# -*- coding: utf-8 -*-
"""
Created on Mon Oct 05 16:22:49 2015

@author: jeromel
"""
import numpy as np
import h5py

def gauss_function(x, a, b0, sigma):
    return a*np.exp(-np.power(x,2)/(2*sigma**2))+b0
    
def extract_avi_from_h5(input_h5_file, output_avi_file):
    """Convert an hdf5-avi file into an avi file
    
    Parameters
    ----------
    output_avi_file : string
    The output avi file path.
    """
    h5_pointer=h5py.File(input_h5_file)
    ds = h5_pointer['movie']
    vid = ds.value
    pad = int(str(ds.dtype).lstrip('S|')) - int(str(vid.dtype).lstrip('S|'))
    
    with open(output_avi_file, 'wb') as f:
        f.write(vid + b'\x00'*pad)
    return