# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 10:38:56 2016
@author: saskiad

"""

import numpy as np
import h5py
import os
from sync import Dataset
import cPickle as pickle
from sklearn.externals import joblib

def Sync_Camera_Stimulus(syncpath, camera):
    '''Computes the alignment of specified camera number with the visual stimulus. 
    Output "frames" is the number of the camera frame during which the stimulus frame (index) occurs'''
    head,tail = os.path.split(syncpath)
    
    d = Dataset(syncpath)
    sample_freq = d.meta_data['ni_daq']['counter_output_freq']
    
    #stimulus frames
    stim_vsync_fall = d.get_falling_edges('stim_vsync')[1:]/sample_freq          #eliminating the DAQ pulse    
    stim_vsync_diff = np.ediff1d(stim_vsync_fall)
    dropped_frames = np.where(stim_vsync_diff>0.033)[0]
    dropped_frames = stim_vsync_fall[dropped_frames]
    long_frames = np.where(stim_vsync_diff>0.1)[0]
    long_frames = stim_vsync_fall[long_frames]
    print "Dropped frames: " + str(len(dropped_frames)) + " at " + str(dropped_frames)
    print "Long frames(>0.1 s): " + str(len(long_frames)) + " at " + str(long_frames) 
    
    try:
        #photodiode transitions
        photodiode_rise = d.get_rising_edges('stim_photodiode')/sample_freq
    
        #test and correct for photodiode transition errors
        ptd_rise_diff = np.ediff1d(photodiode_rise)
        short = np.where(np.logical_and(ptd_rise_diff>0.1, ptd_rise_diff<0.3))[0]
        medium = np.where(np.logical_and(ptd_rise_diff>0.5, ptd_rise_diff<1.5))[0]
        for i in medium:
            if set(range(i-2,i)) <= set(short):
                ptd_start = i+1
            elif set(range(i+1,i+3)) <= set(short):
                ptd_end = i
    
        if ptd_start > 3:
            print "Photodiode events before stimulus start.  Deleted."
        
        ptd_errors = []
        while any(ptd_rise_diff[ptd_start:ptd_end] < 1.8):
            error_frames = np.where(ptd_rise_diff[ptd_start:ptd_end]<1.8)[0] + ptd_start
            print "Photodiode error detected. Number of frames:", len(error_frames)
            photodiode_rise = np.delete(photodiode_rise, error_frames[-1])
            ptd_errors.append(photodiode_rise[error_frames[-1]])
            ptd_end-=1
            ptd_rise_diff = np.ediff1d(photodiode_rise)
            
        #calculate monitor delay
        first_pulse = ptd_start
        delay_rise = np.empty((ptd_end - ptd_start,1))    
        for i in range(ptd_end+1-ptd_start-1):     
            delay_rise[i] = photodiode_rise[i+first_pulse] - stim_vsync_fall[(i*120)+60]
        
        delay = np.mean(delay_rise[:-1])  
        delay_std = np.std(delay_rise[:-1])
        print "Delay:", round(delay, 4)
        print "Delay std:", round(delay_std, 4)
        if delay_std>0.001:
            print "Sync error needs to be fixed"
            delay = 0.0351
            print "Using assumed delay:", round(delay,4)
    except Exception as e:
        print e
        print "Process without photodiode signal"
        delay = 0.0351
        print "Assumed delay:", round(delay, 4)
            
    #adjust stimulus time with monitor delay
    stim_time = stim_vsync_fall + delay
        
    #convert camera1 and camera2 frames into twop frames
    if camera==1:
        print "Synchronizing camera 1"
        cam1_fall = d.get_falling_edges('cam1_exposure')/sample_freq
        frames = np.zeros((len(cam1_fall),1))
        for i in range(len(frames)):
            crossings = np.nonzero(np.ediff1d(np.sign(cam1_fall - stim_time[i]))>0)
            try:
                frames[i] = crossings[0][0]
            except:
                frames[i] = np.NaN    
    
    elif camera==2:        
        print "Synchronizing camera 2"
        cam2_fall = d.get_falling_edges('cam2_exposure')/sample_freq
        frames = np.zeros((len(cam2_fall),1))
        for i in range(len(frames)):
            crossings = np.nonzero(np.ediff1d(np.sign(cam2_fall - stim_time[i]))>0)
            try:
                frames[i] = crossings[0][0]
            except:
                frames[i] = np.NaN
            
    return frames
    
    
def getRunningData(pklpath, frame):
    '''gets running data from stimulus log and downsamples to match camera frames'''
    print "Getting running speed"
    f = open(pklpath, 'rb')
    data = pickle.load(f)
    f.close()
    
    dx = data['items']['foraging']['encoders'][0]['dx']
    vsync_intervals = data['intervalsms']   #in msec
    while len(vsync_intervals)<len(dx):            
        vsync_intervals = np.insert(vsync_intervals, 0, vsync_intervals[0])
    vsync_intervals /= 1000     #in seconds
    if len(dx)==0:
        print "No running data"
    dxcm = ((dx/360)*5.5036*np.pi*2)/vsync_intervals     #converts to cm/s assuming 6.5" wheel with mouse at 2/3 r     
    start = np.nanmin(frame)
    endframe = int(np.nanmax(frame)+1)
    dxds = np.empty((endframe,1))
    for i in range(endframe):
        try:
            temp = np.where(frame==i)[0]
            dxds[i] = np.mean(dxcm[temp[0]:temp[-1]+1])
            if np.isinf(dxds[i]):
                dxds[i] = 0
        except:
            if i<start:
                dxds[i] = np.NaN
            else:
                dxds[i] = dxds[i-1]                             #corrects for dropped frames            
    return dxds   

if __name__=='__main__':
    syncpath = r'/Users/mahdiramadan/Documents/Allen_Institute/code_repository/Data/501021421/501021421_221470_20160128_sync.h5'
    pklpath = r'/Users/mahdiramadan/Documents/Allen_Institute/code_repository/Data/501021421/501021421_221470_20160128_stim.pkl'
    frames = Sync_Camera_Stimulus(syncpath, 2)
    dxds = getRunningData(pklpath, frames)

    frames = joblib.load('frames.pkl')
    dxds = joblib.load('dxds.pkl')
