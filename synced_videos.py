# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:49:10 2015

@author: jeromel
"""
import pandas as pd
import os.path
import video_sync as vs
import numpy as np
import cv2
# import raw_physio_data_flow
# import corr_physio_data_flow
# import aver_physio_data_flow
from raw_behavior import RawBehavior as rb
# import eye_tracking_video_flow
from sync_meta import SyncMeta
# import stimulus_behavior

class SyncedVideos:
    def __init__(self):
        
        # self.raw_physio_data_flow = raw_physio_data_flow
        # self.corr_physio_data_flow = corr_physio_data_flow
        # self.aver_physio_data_flow = aver_physio_data_flow
        self.behavior_data_flow = rb
        # self.eye_tracking_data_flow = eye_tracking_video_flow
        self.sync_meta_flow = SyncMeta
        # self.stimulus_behavior = stimulus_behavior

    def is_valid(self):
        return self.data_present
        
    def get_qc_param(self):
        if hasattr(self,'qc_data'):
            return self.qc_data
        else:
            qc_data = pd.DataFrame()
            
            qc_data['data_minus_sync_nb_frames_physio']=[self.get_nb_physio_dropped_frames()]
            qc_data['data_minus_sync_nb_frames_behavior']=[self.get_nb_beh_dropped_frames()]
            qc_data['data_minus_sync_nb_frames_eye_tracking']=[self.get_nb_eye_dropped_frames()]
            qc_data['data_minus_sync_nb_frames_visual_stim']=[self.stimulus_behavior.get_nb_visual_stim_frames()-self.sync_meta_flow.get_nb_visual_stim_frames()]
    
            # We save the qc internally
            self.qc_data=qc_data
            
            return qc_data       
             
    def get_nb_physio_dropped_frames(self):
        return self.sync_meta_flow.get_nb_physio_frames()-self.raw_physio_data_flow.get_nb_frames()
        
    def get_nb_beh_dropped_frames(self):
        return self.sync_meta_flow.get_nb_behavior_frames()-self.behavior_data_flow.get_nb_frames()
        
    def get_nb_eye_dropped_frames(self):
        return self.sync_meta_flow.get_nb_eye_tracking_frames()-self.eye_tracking_data_flow.get_nb_frames()
                
    def save_qc_param(self,saved_folder):
        self.get_qc_param()        
        file_qc=os.path.join(saved_folder,'synced_videos_qcdata.pkl')
        self.qc_data.to_pickle(file_qc)
        
    def plot_qc(self):
        print("No plots to generate - dataset is not present")
                
    def save_plot_qc(self,saved_folder):
        print("Saving synced video file")
        self.save_synced_video(saved_folder)

    def save_aver_synced_video(self, saved_folder, frame_step = 5, start_frame = 0, max_frame = float('Inf')):
        if self.data_present:
           # We create all SyncedVideoCapture objects
            fname = os.path.join(saved_folder,'aver_videos.avi')

            # We only do it if the file does not exist already
            if not(os.path.exists(fname)):
                # Raw Physio
                h5_data_aver_physio = self.aver_physio_data_flow.data_pointer['data']
                
                timing_raw_physio = self.sync_meta_flow.get_frame_times_physio(corrected = True)
                
                timing_aver_physio = timing_raw_physio[::8]
                
                # We cut the last timing. Downsample discards unfinished blocks
                if len(timing_aver_physio)!=len(timing_raw_physio)/8.0:
                    timing_aver_physio = timing_aver_physio[0:len(timing_aver_physio)-1]
                    
                synced_aver_physio = vs.SyncedVideoCapture(h5_data_aver_physio, timing_aver_physio)
    
                # Behavior
                self.behavior_data_flow.load_data_pointer()
                data_behavior = self.behavior_data_flow.tmp_file_string
                timing_behavior = self.sync_meta_flow.get_frame_times_behavior(corrected = True)
                
                synced_behavior = vs.SyncedVideoCapture(data_behavior, timing_behavior)
                    
                # Eye tracking
                self.eye_tracking_data_flow.load_data_pointer()
                data_eye_tracking = self.eye_tracking_data_flow.tmp_file_string
                timing_eye_tracking = self.sync_meta_flow.get_frame_times_eye_tracking(corrected = True)
                
                synced_eye_tracking = vs.SyncedVideoCapture(data_eye_tracking, timing_eye_tracking)
                    
                all_videos = vs.MovieCombiner(frame_step = frame_step, out_color = True)
                
                top_aver_physio_range = self.raw_physio_data_flow.get_quantile_movie(99)
                bottom_aver_physio_range = self.raw_physio_data_flow.get_quantile_movie(1)
                
                top_behav_range = self.behavior_data_flow.get_quantile_movie(100)
                bottom_behav_range = self.behavior_data_flow.get_quantile_movie(0)
                
                top_eye_tracking_range = self.eye_tracking_data_flow.get_quantile_movie(100)
                bottom_eye_tracking_range = self.eye_tracking_data_flow.get_quantile_movie(0)
                
                def scale_picture(tmp, top_clip, bottom_clip, red_saturation):
    
#                    xy_coords = np.where(tmp == red_saturation)
                    tmp = np.clip(tmp, bottom_clip, top_clip)
                    tmp = (tmp.astype('float')-bottom_clip)/(top_clip-bottom_clip)
                    bottom_final = 0
                    top_final = 255
                    tmp = tmp*(top_final-bottom_final)+bottom_final
                    tmp =  tmp.astype('uint8')    
    
                    if tmp.ndim == 2:
                        tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
                    
                    # We don't do saturation display for this movie
#                    tmp[xy_coords[0], xy_coords[1], 0] = 0
#                    tmp[xy_coords[0], xy_coords[1], 1] = 0
#                    tmp[xy_coords[0], xy_coords[1], 2] = 255
    
                    return tmp
                    
                def scale_picture_physio(tmp): 
                    return scale_picture(tmp, top_aver_physio_range, bottom_aver_physio_range, 4095) 
                def scale_picture_behavior(tmp): 
                    return scale_picture(tmp, top_behav_range, bottom_behav_range, 255)
                def scale_picture_eye_tracking(tmp): 
                    return scale_picture(tmp, top_eye_tracking_range, bottom_eye_tracking_range, 255)
                    
                all_videos.add_stream(synced_behavior, 1, 1, frame_callback = scale_picture_behavior)
                all_videos.add_stream(synced_aver_physio, 2, 1, frame_callback = scale_picture_physio)
                all_videos.add_stream(synced_eye_tracking, 3, 1, frame_callback = scale_picture_eye_tracking)

                all_videos.write_movie(fname, start_frame = start_frame, max_frame = max_frame, add_frame_nb = False)      
        
        
    def save_synced_video(self,saved_folder, start_frame = 0, max_frame = float('Inf')):
        if self.data_present:
            # We create all SyncedVideoCapture objects
            fname = os.path.join(saved_folder,'all_videos.avi')

            # We only do it if the file does not exist already
            if not(os.path.exists(fname)):
                # Raw Physio
                h5_data_raw_physio = self.raw_physio_data_flow.data_pointer['data']
                timing_raw_physio = self.sync_meta_flow.get_frame_times_physio(corrected = True)
                
                synced_raw_physio = vs.SyncedVideoCapture(h5_data_raw_physio, timing_raw_physio)
    
                # Corr Physio
                h5_data_corr_physio = self.corr_physio_data_flow.data_pointer['data']
                
                synced_corr_physio = vs.SyncedVideoCapture(h5_data_corr_physio, timing_raw_physio)
    
                # Behavior
                self.behavior_data_flow.load_data_pointer()
                data_behavior = self.behavior_data_flow.tmp_file_string
                timing_behavior = self.sync_meta_flow.get_frame_times_behavior(corrected = True)
                
                synced_behavior = vs.SyncedVideoCapture(data_behavior, timing_behavior)
                    
                # Eye tracking
                self.eye_tracking_data_flow.load_data_pointer()
                data_eye_tracking = self.eye_tracking_data_flow.tmp_file_string
                timing_eye_tracking = self.sync_meta_flow.get_frame_times_eye_tracking(corrected = True)
                
                synced_eye_tracking = vs.SyncedVideoCapture(data_eye_tracking, timing_eye_tracking)
                    
                all_videos = vs.MovieCombiner(frame_step = 10, out_color = True)
                
                top_raw_physio_range = self.raw_physio_data_flow.get_quantile_movie(99)
                bottom_raw_physio_range = self.raw_physio_data_flow.get_quantile_movie(1)
                top_behav_range = self.behavior_data_flow.get_quantile_movie(100)
                bottom_behav_range = self.behavior_data_flow.get_quantile_movie(0)
                top_eye_tracking_range = self.eye_tracking_data_flow.get_quantile_movie(100)
                bottom_eye_tracking_range = self.eye_tracking_data_flow.get_quantile_movie(0)
                
                def scale_picture(tmp, top_clip, bottom_clip, red_saturation):
    
                    xy_coords = np.where(tmp == red_saturation)
                    tmp = np.clip(tmp, bottom_clip, top_clip)
                    tmp = (tmp.astype('float')-bottom_clip)/(top_clip-bottom_clip)
                    bottom_final = 0
                    top_final = 255
                    tmp = tmp*(top_final-bottom_final)+bottom_final
                    tmp =  tmp.astype('uint8')    
    
                    if tmp.ndim == 2:
                        tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
                    
                    tmp[xy_coords[0], xy_coords[1], 0] = 0
                    tmp[xy_coords[0], xy_coords[1], 1] = 0
                    tmp[xy_coords[0], xy_coords[1], 2] = 255
    
                    return tmp
                    
                def scale_picture_physio(tmp): 
                    return scale_picture(tmp, top_raw_physio_range, bottom_raw_physio_range, 4095) 
                def scale_picture_behavior(tmp, value):
                    return scale_picture(tmp, top_behav_range, bottom_behav_range, value)
                def scale_picture_eye_tracking(tmp): 
                    return scale_picture(tmp, top_eye_tracking_range, bottom_eye_tracking_range, 255)
                    
                all_videos.add_stream(synced_raw_physio, 1, 1, frame_callback = scale_picture_physio)
                all_videos.add_stream(synced_corr_physio, 1, 2, frame_callback = scale_picture_physio)
                all_videos.add_stream(synced_behavior, 2, 1, frame_callback = scale_picture_behavior)
                all_videos.add_stream(synced_eye_tracking, 2, 2, frame_callback = scale_picture_eye_tracking)
                
                all_videos.write_movie(fname, start_frame = start_frame, max_frame = max_frame)

    def video_annotation(self, exp_folder):
        # outputs a .mp4 video, sped up x 2, with displayed frame count in upper right.
        file_name = rb(exp_folder).get_file_string()
        data_pointer = cv2.VideoCapture(file_name)
        fps = data_pointer.get(cv2.cv.CV_CAP_PROP_FPS)
        nFrames = int(data_pointer.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        frameWidth = int(data_pointer.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        frameHeight = int(data_pointer.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.cv.CV_FOURCC(*'mp4v')
        #change 3rd parameter of out function for different playback speeds
        out = cv2.VideoWriter('output.mp4', fourcc, fps*2, (frameWidth, frameHeight))
        ret, frame = data_pointer.read()

        while ret:

            frame_count = int(data_pointer.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
            cv2.putText(img=frame,
                        text=str(frame_count),
                        org=(480, 130),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=2,
                        color=(0, 255, 0),
                        thickness=2,
                        lineType=cv2.CV_AA)

            out.write(frame)
            ret, frame = data_pointer.read()

