# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:49:10 2015

@author: jeromel
"""
import pandas as pd
import os.path
from sync_lib import Dataset
import numpy as np
import matplotlib.pylab as plt

class SyncMeta:
    def __init__(self, data_folder):      
        self.data_present = False

        for file in os.listdir(data_folder):
            if file.endswith("_sync.h5"):
                # pointer to sync file
                self.file_string=os.path.join(data_folder,file)
                self.data_pointer=Dataset(self.file_string)
                self.data_present=True

    def is_valid(self):
        return self.data_present
        
    def get_qc_param(self):
        if hasattr(self,'qc_data'):
            print("No qc is available")                   
            return self.qc_data
        else:
            qc_data = pd.DataFrame()
            
            qc_data['nb_dropped_frames_visual_stim']=[self.get_nb_dropped_frames_visual_stim()]
            qc_data['sampl_freq']=[self.get_sampling_freq()]
            
            # Commenting as temporally not working on Scientifica
#            qc_data['nb_sweeps']=[self.get_nb_sweep()]
            qc_data['nb_frame_physio']=[self.get_nb_physio_frames()]
            qc_data['nb_frame_behavior']=[self.get_nb_behavior_frames()]
            qc_data['nb_frame_eye_tracking']=[self.get_nb_eye_tracking_frames()]
            qc_data['nb_frame_visual_stim']=[self.get_nb_visual_stim_frames()]
            qc_data['mean_rate_physio']=[self.get_mean_rate_physio()]
            qc_data['mean_rate_behavior']=[self.get_mean_rate_behavior()]
            qc_data['mean_rate_eye_tracking']=[self.get_mean_rate_eye_tracking()]
            qc_data['mean_rate_visual_stim']=[self.get_mean_rate_visual_stim()]
            qc_data['std_period_physio']=[self.get_std_period_physio()]
            qc_data['std_period_eye_tracking']=[self.get_std_period_eye_tracking()]
            qc_data['std_period_behavior']=[self.get_std_period_behavior()]
            qc_data['std_period_visual_stim']=[self.get_std_period_visual_stim()]
            qc_data['physio_duration_min']=[self.get_physio_duration_min()]
            qc_data['eye_tracking_duration_min']=[self.get_eye_tracking_duration_min()]
            qc_data['behavior_duration_min']=[self.get_behavior_duration_min()]
            qc_data['visual_stim_duration_min']=[self.get_visual_stim_duration_min()]
 
           # We save the qc internally
            self.qc_data=qc_data
            
            return qc_data   
            
    def get_sampling_freq(self):
        return self.data_pointer.meta_data['ni_daq']['counter_output_freq']

    def get_nb_physio_frames(self):
        return len(self.get_frame_times_physio())

    def get_nb_behavior_frames(self):
        return len(self.get_frame_times_behavior())
        
    def get_nb_sweep(self):
        return len(self.get_sweep_start_times())
                
    def get_nb_eye_tracking_frames(self):
        return len(self.get_frame_times_eye_tracking())
        
    def get_nb_visual_stim_frames(self):
        return len(self.get_frame_times_visual_stim())
        
    def get_mean_rate_physio(self):
        return 1.0/np.diff(self.get_frame_times_physio()).mean()

    def get_mean_rate_behavior(self):
        return 1.0/np.diff(self.get_frame_times_behavior()).mean()

    def get_mean_rate_eye_tracking(self):
        return 1.0/np.diff(self.get_frame_times_eye_tracking()).mean()

    def get_mean_rate_visual_stim(self):
        return 1.0/np.diff(self.get_frame_times_visual_stim()).mean()
        
    def get_frame_times_physio(self, corrected = False):
        
        all_times = self.get_frame_times('2p_vsync', slope = 'falling')
    
        # TODO : findout why we have diff nb of frames in hdf5 file        
        if corrected and len(all_times)>0:
            all_times = all_times[1:]
        return all_times
    
    def get_frame_times_behavior(self, corrected = False):
        all_times = self.get_frame_times('cam1_exposure', slope = 'rising')  
        
        # TODO : findout why we have diff nb of frames in avi file     
        if corrected and len(all_times)>0:
            all_times = np.append(all_times, all_times[-1]+0.0001)
        return all_times       
     
    def get_frame_times_eye_tracking(self, corrected = False):
        all_times = self.get_frame_times('cam2_exposure', slope = 'rising')  
        
        # TODO : findout why we have diff nb of frames in avi file  
        if corrected and len(all_times)>0:
            all_times = np.append(all_times, all_times[-1]+0.0001)                                
        return all_times 
        
    def get_frame_times_led(self):
        all_times = self.get_frame_times('stim_photodiode', slope = 'rising')
        
        return all_times
        

    def get_sweep_start_times(self):
        all_times = self.get_frame_times('stim_sweep', slope = 'rising')  

        return all_times
        
    def get_sweep_end_times(self):
        all_times = self.get_frame_times('stim_sweep', slope = 'falling')  

        return all_times      
        
    def get_visual_stim_duration_min(self):
        all_times = self.get_frame_times_visual_stim()
        duration = (all_times[-1]-all_times[0])/60
        
        return duration

    def get_behavior_duration_min(self):
        all_times = self.get_frame_times_behavior()
        
        if len(all_times)==0:
            duration = 0
        else:
            duration = (all_times[-1]-all_times[0])/60
        
        return duration

    def get_physio_duration_min(self):
        all_times = self.get_frame_times_visual_stim()
        duration = (all_times[-1]-all_times[0])/60
        
        return duration

    def get_eye_tracking_duration_min(self):
        all_times = self.get_frame_times_eye_tracking()
        
        if len(all_times)==0:
            duration = 0
        else:
            duration = (all_times[-1]-all_times[0])/60
        
        return duration
        
    def get_frame_times_visual_stim(self, corrected = False):
        # stimulus vsyncs
        vs_r = self.data_pointer.get_rising_edges('stim_vsync')
        vs_f = self.data_pointer.get_falling_edges('stim_vsync')
               
        sample_freq=self.get_sampling_freq()

        # convert to seconds
        vs_r_sec = vs_r/sample_freq
        vs_f_sec = vs_f/sample_freq
        
        #   NOTICE THE BRIEF SPIKE THAT OCCURS INITIALLY AT 216.57 SECONDS.  THIS IS HAPPENING WHEN I AM INITIALIZING THE
        #   DAQ AND I'M LOOKING INTO RESOLVING THIS. FOR NOW WE IGNORE IT IF IT EXISTS:
        
        if (corrected and vs_r_sec[1] - vs_r_sec[0] > 0.5):
            vsyncs = vs_f_sec[1:]
        else:
            vsyncs = vs_f_sec

        return vsyncs
        
    def get_nb_dropped_frames_visual_stim(self):               
        # frame intervals
        vsyncs = self.get_frame_times_visual_stim(corrected = True)
        vs_ints = np.diff(vsyncs)
        
        return len(vsyncs[np.where(vs_ints>0.025)])

    def get_nb_dropped_photodiodes(self):               
        all_times = self.get_frame_times_led()
        diff_time_matrix = np.diff(all_times)
        all_large = diff_time_matrix>2.3
        all_small = diff_time_matrix<1.7
        excessive_delays = diff_time_matrix[np.where(np.any([all_large, all_small],0))]
        return len(excessive_delays)
#    
#    def get_monitor_delay(self):
#        first_pulse = ptd_start
#        delay_rise = np.empty((ptd_end - ptd_start,1))  
#        stim_vsync_fall = self.get_frame_times_visual_stim(corrected = True)
#        
#        for i in range(ptd_end-ptd_start):     
#            delay_rise[i] = photodiode_rise[i+first_pulse] - stim_vsync_fall[(i*120)+60]
#       
#        return delay_rise
#        
#    delay = np.mean(delay_rise[:-1])   
        
        
    def get_std_period_physio(self):
        vsyncs = self.get_frame_times_physio(corrected = False)
        return np.std(np.diff(vsyncs))
        
    def get_std_period_eye_tracking(self):
        vsyncs = self.get_frame_times_eye_tracking(corrected = False)
        return np.std(np.diff(vsyncs))
        
    def get_std_period_behavior(self):
        vsyncs = self.get_frame_times_behavior(corrected = True)
        return np.std(np.diff(vsyncs))
        
    def get_std_period_visual_stim(self):
        vsyncs = self.get_frame_times_visual_stim(corrected = True)
        return np.std(np.diff(vsyncs))

    def plot_dist_period_physio(self):
        all_times = self.get_frame_times_physio()
        return self.plot_dist_period(all_times)
    
    def plot_dist_period_behavior(self):
        all_times = self.get_frame_times_behavior()
        return self.plot_dist_period(all_times)        
     
    def plot_dist_period_eye_tracking(self):
        all_times = self.get_frame_times_eye_tracking()
        return self.plot_dist_period(all_times)

    def plot_dist_period_visual_stim(self):
        all_times = self.get_frame_times_visual_stim()
        return self.plot_dist_period(all_times)
                                
    def plot_dist_period(self, all_times):            
        
        fig1=plt.figure()

        if len(all_times)>0:
            periods = np.diff(all_times)*1000        
            plt.hist(periods, bins=100)
            plt.title("Frame interval histogram")
            plt.xlabel("time interval (ms)")      

        return fig1

    def plot_eye_diff_frames(self):
        all_times = self.get_frame_times_eye_tracking()
        
        diff_time_matrix = np.diff(all_times)
        
        fig1=plt.figure()
        plt.plot(diff_time_matrix)
        plt.xlabel("Frame number")
        plt.ylabel("Frame delta (s)")
        plt.title('Eye movie')
        plt.axes
        plt.ylim(0.020,0.045)

        return fig1

    def plot_physio_diff_frames(self):
        all_times = self.get_frame_times_physio()
        
        diff_time_matrix = np.diff(all_times)
        
        fig1=plt.figure()
        plt.plot(diff_time_matrix)
        plt.xlabel("Frame number")
        plt.ylabel("Frame delta (s)")
        plt.title('Physio movie')
        plt.axes
        plt.ylim(0.020,0.045)

        return fig1        
        
    def plot_stim_diff_frames(self):
        all_times = self.get_frame_times_visual_stim()
        
        diff_time_matrix = np.diff(all_times)
        
        fig1=plt.figure()
        plt.plot(diff_time_matrix)
        plt.xlabel("Frame number")
        plt.ylabel("Frame delta (s)")
        plt.title('Stim movie')
        plt.axes
        plt.ylim(0.010,0.025)

        return fig1        

    def plot_behavior_diff_frames(self):
        all_times = self.get_frame_times_behavior()
        
        diff_time_matrix = np.diff(all_times)
        
        fig1=plt.figure()
        plt.plot(diff_time_matrix)
        plt.xlabel("Frame number")
        plt.ylabel("Frame delta (s)")
        plt.title('Behavior movie')
        plt.axes
        plt.ylim(0.020,0.045)

        return fig1
        
    def plot_photodiode_diff_frames(self):
        all_times = self.get_frame_times_led()
        
        diff_time_matrix = np.diff(all_times)
        
        fig1=plt.figure()
        plt.plot(diff_time_matrix)
        plt.xlabel("Frame number")
        plt.ylabel("Photodiode delta (s)")
        plt.title('Photodiode changes')
        plt.axes
        plt.ylim(1,3)
        
        return fig1
        
    def get_frame_times(self, data_string, slope='falling'):
        if slope=='rising':
            begin_frames_timing = self.data_pointer.get_rising_edges(data_string)
        else:
            begin_frames_timing = self.data_pointer.get_falling_edges(data_string)

        sample_freq=self.get_sampling_freq()
        
        #Convert to seconds
        begin_frames_timing_s = begin_frames_timing/sample_freq

        return begin_frames_timing_s
        
    def save_qc_param(self,saved_folder):
        self.get_qc_param()        
        file_qc=os.path.join(saved_folder,'sync_meta_qcdata.pkl')
        self.qc_data.to_pickle(file_qc)              
        
    def plot_qc(self):
        if self.data_present:
            try:
                self.plot_dist_period_physio()
                self.plot_dist_period_behavior()
                self.plot_dist_period_eye_tracking()
                self.plot_dist_period_visual_stim()                
            except:
                print("Error generating plots")
                
    def save_plot_qc(self,saved_folder):
        if self.data_present:
            try:
                dist_period_physio_png=os.path.join(saved_folder,'dist_period_physio.png')
                fig=self.plot_dist_period_physio()
                fig.savefig(dist_period_physio_png)
                plt.close(fig)
                
                dist_period_behavior_png=os.path.join(saved_folder,'dist_period_behavior.png')
                fig=self.plot_dist_period_behavior()
                fig.savefig(dist_period_behavior_png)
                plt.close(fig)

                dist_period_eye_tracking_png=os.path.join(saved_folder,'dist_period_eye_tracking.png')
                fig=self.plot_dist_period_eye_tracking()
                fig.savefig(dist_period_eye_tracking_png)
                plt.close(fig)

                dist_period_visual_stim_png=os.path.join(saved_folder,'dist_period_visual_stim.png')
                fig=self.plot_dist_period_visual_stim()
                fig.savefig(dist_period_visual_stim_png)
                plt.close(fig)

                diff_period_eye_png=os.path.join(saved_folder,'diff_period_eye.png')
                fig=self.plot_eye_diff_frames()
                fig.savefig(diff_period_eye_png)
                plt.close(fig)

                diff_period_behavior_png=os.path.join(saved_folder,'diff_period_behavior.png')
                fig=self.plot_behavior_diff_frames()
                fig.savefig(diff_period_behavior_png)
                plt.close(fig)                
 
                diff_period_physio_png=os.path.join(saved_folder,'diff_period_physio.png')
                fig=self.plot_physio_diff_frames()
                fig.savefig(diff_period_physio_png)
                plt.close(fig)               

                diff_period_stim_png=os.path.join(saved_folder,'diff_period_stim.png')
                fig=self.plot_stim_diff_frames()
                fig.savefig(diff_period_stim_png)
                plt.close(fig)           

                diff_photodiode_png=os.path.join(saved_folder,'diff_photodiode.png')
                fig=self.plot_photodiode_diff_frames()
                fig.savefig(diff_photodiode_png)
                plt.close(fig)  
                     
            except:
                print("Error generating plots")
            


   

        



