# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:49:10 2015

@author: jeromel
"""
import pandas as pd
import pickle 
import os
import os.path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime 

class StimulusBehavior:
    def __init__(self, exp_folder):         
        
        self.file_string=''
        for file in os.listdir(exp_folder):
            if file.endswith("_stim.pkl"):                
                # We check if the file is accessible but we do not load it
                self.file_string=os.path.join(exp_folder,file)

        if os.path.isfile(self.file_string):
            self.data_present = True
        else:
            self.data_present = False
        
    def is_valid(self):
        return self.data_present
        
    def get_qc_param(self):
        if not(hasattr(self,'qc_data')):
            qc_data = pd.DataFrame()
            
            qc_data['mean_mouse_speed']=[self.get_mean_mouse_speed()]
            qc_data['total_travelled_distance']=[self.get_total_travelled_distance()]
#            qc_data['nb_sweeps']=[self.get_nb_sweeps()]
#            qc_data['nb_unique_stimuli']=[self.get_nb_unique_stimuli()]
            qc_data['script_path']=[self.get_script_path()]
            qc_data['script_filename']=[self.get_script_filename()]
            qc_data['stimulus_type']=[self.get_stimulus_type()]
#            qc_data['start_date']=[self.get_start_date()]
#            qc_data['stop_date']=[self.get_stop_date()]
#            qc_data['swept_param_values']=[self.get_swept_param_values()]
#            qc_data['swept_param_names']=[self.get_swept_param_names()]
#            qc_data['stim_duration_seconds']=[self.get_stim_duration_seconds()]
            qc_data['nb_frame_visual_stim']=[self.get_nb_visual_stim_frames()]

            # We save the qc internally
            self.qc_data=qc_data
            
        return self.qc_data    
                
    def load_data_pointer(self):
        if not(hasattr(self,'data_pointer')):
            opened_file=open(self.file_string)
            self.data_pointer=pickle.load(opened_file)
            opened_file.close()
              
    def save_qc_param(self,saved_folder):
        self.get_qc_param()        
        file_qc=os.path.join(saved_folder,'stimulus_behavior_qcdata.pkl')
        self.qc_data.to_pickle(file_qc)    
        
    def plot_qc(self):
        if self.data_present:
            try:
                self.plot_distrib_mouse_motion()
                self.plot_mouse_walk()
            except:
                print("Error generating plots")
                
    def save_plot_qc(self,saved_folder):
        if self.data_present:
            try:
                distrib_mouse_motion_png=os.path.join(saved_folder,'distrib_mouse_motion.png')
                fig=self.plot_distrib_mouse_motion()
                fig.savefig(distrib_mouse_motion_png)
                
                mouse_walk_png=os.path.join(saved_folder,'mouse_walk.png')
                fig=self.plot_mouse_walk()
                fig.savefig(mouse_walk_png)
 
                mouse_wheel_png=os.path.join(saved_folder,'mouse_raw_wheel.png')
                fig=self.plot_raw_mouse_wheel()
                fig.savefig(mouse_wheel_png)
                
            except:
                print("Error generating plots")
                
    def get_mean_mouse_speed(self):
        self.load_data_pointer()
        dx_data=self.data_pointer['items']['foraging']['encoders'][0]['dx']
        dx_data_cm = ((dx_data/360)*5.5036*np.pi*2)/(1/self.data_pointer['fps'])

        mean_speed=dx_data_cm.flatten().mean()
        
        return mean_speed
        
    def get_script_filename(self):
        self.load_data_pointer()
        return os.path.split(self.data_pointer['script'])[1]

    def get_script_text(self):
        self.load_data_pointer()
        return self.data_pointer['scripttext']
    
    def get_stimulus_type(self):
        import re
        all_stim_text = self.get_script_text()
        r = re.compile('"""\n(.*?).py')
        stim_name = r.search(all_stim_text)
        if stim_name:
            stim_name = stim_name.group(1)
        
        return stim_name
    
    def get_nb_sweeps(self):
        return len(self.get_presented_sweep_table())
        
    def get_nb_visual_stim_frames(self):
        self.load_data_pointer()
        return int(self.data_pointer['vsynccount'])
        
    def get_nb_unique_stimuli(self):
        return len(self.get_ordered_sweep_table())
        
    def get_script_path(self):
        self.load_data_pointer()
        return self.data_pointer['script']
        
    def get_start_date(self):
        self.load_data_pointer()
        x = self.data_pointer['startdatetime']
        return x

    def get_stop_date(self):
        self.load_data_pointer()
        x= self.data_pointer['stopdatetime']
        return x

    def get_stim_duration_seconds(self):
        #duration is precise down to +-0.99 seconds (due to .frames at end of seconds)
        duration = self.get_stop_date()-self.get_start_date()
        return float(duration.seconds)
        
    def get_swept_param_values(self):
        self.load_data_pointer()
        return str(self.data_pointer['bgSweep'])
        
    def get_swept_param_names(self):
        self.load_data_pointer()
        return str(self.data_pointer['bgdimnames'])

    def get_ordered_sweep_table(self):
        self.load_data_pointer()
        return self.data_pointer['bgsweeptable']
        
    def get_presented_sweep_table(self):
        self.load_data_pointer()
        return [self.data_pointer['bgsweeptable'][i] for i in self.data_pointer['bgsweeporder']]
        
    def get_total_travelled_distance(self):
        self.load_data_pointer()
        walking_distance=self.data_pointer['items']['foraging']['encoders'][0]['dx'].sum()

        return walking_distance

    def plot_distrib_mouse_motion(self, range_plot=[-10,10]):
        self.load_data_pointer()
        fig1=plt.figure()
        dx_data=self.data_pointer['items']['foraging']['encoders'][0]['dx']
        dx_data=np.clip(dx_data.flatten(),range_plot[0],range_plot[1])
        plt.hist(dx_data, bins=100, range=range_plot, normed=True)
        plt.xlabel("Mouse speed (au)")
        plt.xlim(range_plot[0],range_plot[1])
        plt.ylabel("pdf")
        
        return fig1
                            
    def plot_mouse_walk(self):
        self.load_data_pointer()
        
        dx_data=self.data_pointer['items']['foraging']['encoders'][0]['dx']
        dx_data_cm = ((dx_data/360)*5.5036*np.pi*2)/(1/self.data_pointer['fps'])
        
        walking_distance=dx_data_cm.cumsum()*(1/self.data_pointer['fps'])
        plt.figure()
        plt.plot(walking_distance/100)
        plt.xlabel("Frames")
        plt.ylabel("Traveled distance (m)")
        fig1=plt.gcf()      
        
        return fig1
        
    def plot_raw_mouse_wheel(self):
        self.load_data_pointer()
        walking_speed=self.data_pointer['items']['foraging']['encoders'][0]['dx']
        walking_speed_cm = ((walking_speed/360)*5.5036*np.pi*2)/(1/self.data_pointer['fps'])

        plt.figure()
        plt.plot(walking_speed_cm)
        plt.xlabel("Frames")
        plt.ylabel("wheel speed (cm/s)")
        fig1=plt.gcf()
        return fig1

    def raw_mouse_wheel(self):
        # returns ndarray of the raw wheel rotation data
        self.load_data_pointer()
        walking_speed = self.data_pointer['items']['foraging']['encoders'][0]['dx']
        walking_speed_cm = ((walking_speed / 360) * 5.5036 * np.pi * 2) / (1 / self.data_pointer['fps'])

        return walking_speed_cm