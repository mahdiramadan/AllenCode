"""
Created on Tue Oct 20 13:49:10 2015

@author: jeromel
"""
import pandas as pd
import os
import ophyse.mathfcn as mf
import matplotlib.pyplot as plt
import numpy as np 
import cv2
from PIL import Image, ImageDraw

class RawBehavior:
    def __init__(self, exp_folder, tmp_folder='C:\\Tmp\\'):         
        
        self.file_string=''
        self.tmp_folder=tmp_folder
        
        for file in os.listdir(exp_folder):
            if file.endswith(".avi"):
                # We check if the file is accessible but we do not load it
                self.file_string=os.path.join(exp_folder,file)
                tmp_filename=os.path.splitext(file)[0]
                file_string_avi = os.path.join(exp_folder,tmp_filename)
                
                if os.path.isfile(file_string_avi):
                    self.tmp_file_string = file_string_avi
                    self.data_pointer = cv2.VideoCapture(self.tmp_file_string)
                    self.delete_tmp = False
                else:
                    self.tmp_file_string=os.path.join(self.tmp_folder,tmp_filename)
                    self.delete_tmp = True

        if os.path.isfile(self.file_string):
            self.data_present = True
        else:
            self.data_present = False
     
    def __del__(self):
        # We remove temporary files if present
        if hasattr(self,'data_pointer'):
            self.data_pointer=[]
            if self.delete_tmp:
                os.remove(self.tmp_file_string)         
        
    def is_valid(self):
        return self.data_present
        
    def get_qc_param(self):
        if not(hasattr(self,'qc_data')):
            qc_data = pd.DataFrame()
            
            qc_data['mean_movie']=[self.get_mean_movie()]
            qc_data['median_movie']=[self.get_median_movie()]

            qc_data['nb_frames']=[self.get_nb_frames()]             
    
            qc_data['max_movie']=[self.get_max_movie()]
            qc_data['nb_pixel_sat']=[self.get_nb_pixel_sat()]
            qc_data['95_quantile_movie']=[self.get_quantile_movie(95)]
            qc_data['5_quantile_movie']=[self.get_quantile_movie(5)]
            
            # We save the qc internally
            self.qc_data=qc_data
            
        return self.qc_data  
        
    def plot_qc(self):
        if self.data_present:
            try:
                self.plot_distrib_hist()
                self.plot_single_image()
            except:
                print("Error generating plots")
                         
    def save_qc_param(self,saved_folder):
        self.get_qc_param()        
        file_qc=os.path.join(saved_folder,'raw_behavior_qcdata.pkl')
        self.qc_data.to_pickle(file_qc)              
                
    def save_plot_qc(self,saved_folder):
        if self.data_present:
            try:
                distrib_hist_png=os.path.join(saved_folder,'distrib_hist.png')
                fig=self.plot_distrib_hist()
                fig.savefig(distrib_hist_png)
                plt.close(fig)

                single_image_max_png=os.path.join(saved_folder,'single_image_max.png')
                fig=self.plot_single_image(scaling='max')
                fig.savefig(single_image_max_png)
                plt.close(fig)

                single_image_quantile_png=os.path.join(saved_folder,'single_image_quantile.png')
                fig=self.plot_single_image(scaling='quantile')
                fig.savefig(single_image_quantile_png)
                plt.close(fig)

                begin_image_full_png=os.path.join(saved_folder,'begin_image_full.png')
                fig=self.plot_single_image(frame_nb=500, scaling='full')
                fig.savefig(begin_image_full_png)
                plt.close(fig)
                
                end_image_full_png=os.path.join(saved_folder,'end_image_full.png')
                fig=self.plot_single_image(frame_nb=self.get_nb_frames()-500, scaling='full')
                fig.savefig(end_image_full_png)
                plt.close(fig)
                
                single_image_full_png=os.path.join(saved_folder,'single_image_full.png')
                fig=self.plot_single_image(scaling='full')
                fig.savefig(single_image_full_png)
                plt.close(fig)

                norm_var_image_png=os.path.join(saved_folder,'norm_var_image.png')
                fig=self.plot_var_image()
                fig.savefig(norm_var_image_png)   
                plt.close(fig)
            except:
                print("Error generating plots")
                               
    def load_data_pointer(self):
        if not(hasattr(self,'data_pointer')):
            mf.extract_avi_from_h5(self.file_string, self.tmp_file_string)
            self.data_pointer = cv2.VideoCapture(self.tmp_file_string)
                   
    def get_mean_movie(self,nb_frames_to_use = 500):
        movie_subset = self.get_movie_sub_block(nb_frames_to_use=nb_frames_to_use)
        
        return movie_subset.flatten().mean()

    def get_xy_size(self):
        if not(hasattr(self,'xy_size')):
            local_image = self.get_image(frame_nb=0)
            self.xy_size = local_image.shape
        return self.xy_size
                
    def get_nb_frames(self):
        if self.data_present:        
            self.load_data_pointer()
            N_frames = self.data_pointer.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
            return int(N_frames)
        else:
            return 0  
            
    def get_quantile_movie(self, quantile, nb_frames_to_use = 500):
        if self.data_present:
            local_data = self.get_movie_sub_block(nb_frames_to_use=nb_frames_to_use)
            local_percentile = np.percentile(local_data.flatten(),quantile)    
            return local_percentile
        else:
            return np.nan
            
    def get_median_movie(self,nb_frames_to_use = 500):
        movie_subset = self.get_movie_sub_block(nb_frames_to_use=nb_frames_to_use)
        
        return np.median(movie_subset.flatten())

    def get_max_movie(self,nb_frames_to_use = 500):
        movie_subset = self.get_movie_sub_block(nb_frames_to_use=nb_frames_to_use)
        
        return movie_subset.flatten().max()
        
    def get_nb_pixel_sat(self,nb_frames_to_use = 500):
        movie_subset = self.get_movie_sub_block(nb_frames_to_use=nb_frames_to_use)

        local_data = movie_subset[0:nb_frames_to_use].max(axis = 0).flatten() == 255
        total_sat = local_data.sum()
        
        return total_sat
                     
    def get_rgb_image(self, frame_nb = 500):
        self.load_data_pointer()
        
        self.data_pointer.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_nb)
        rc, eye_tracking_image = self.data_pointer.read()
                
        return eye_tracking_image
        
    def get_image(self, frame_nb=500):
        self.load_data_pointer()
        
        self.data_pointer.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frame_nb)
        rc, eye_tracking_image = self.data_pointer.read()
        
        eye_tracking_image=np.mean(eye_tracking_image, axis = 2)
        
        return eye_tracking_image
        
    def get_movie_sub_block(self, nb_frames_to_use=500):
        self.load_data_pointer()

        if not(hasattr(self,'nb_frames_tmp_subset')) or self.nb_frames_tmp_subset != nb_frames_to_use:
            self.nb_frames_tmp_subset=nb_frames_to_use
            final_size=list(self.get_image(frame_nb = 0).shape)
            final_size.append(nb_frames_to_use)
            movie_subset=np.zeros(final_size)
            for index in range(0,nb_frames_to_use):
                movie_subset[:,:,index]=self.get_image(frame_nb = index)
            
            self.tmp_subset = movie_subset
        else:
            movie_subset = self.tmp_subset
            
        return movie_subset
            
    def plot_distrib_hist(self,nb_frames_to_use=500, range_plot=[0,255]):
        movie_subset = self.get_movie_sub_block(nb_frames_to_use=nb_frames_to_use)
        fig1=plt.figure()
        movie_subset=np.clip(movie_subset.flatten(),range_plot[0],range_plot[1])
        plt.hist(movie_subset, bins=100, range=range_plot, normed=True)
        plt.xlabel("Pixel value")
        plt.xlim(range_plot[0],range_plot[1])
        plt.ylabel("pdf")
        

        return fig1
        
    def plot_single_image(self, frame_nb=500, scaling='max'):
        eye_tracking_image = self.get_image(frame_nb = frame_nb)

        fig1=plt.figure()
        if scaling == 'max': 
            plt.imshow(eye_tracking_image, cmap = 'gray')
        elif scaling == 'quantile':
            bottom_scale=np.percentile(eye_tracking_image,5)            
            top_scale=np.percentile(eye_tracking_image,95)
            plt.imshow(eye_tracking_image, cmap = 'gray', clim = [bottom_scale, top_scale])
        elif scaling == 'full':
            bottom_scale=0           
            top_scale=255
            plt.imshow(eye_tracking_image, cmap = 'gray', clim = [bottom_scale, top_scale])
  
        
        
        return fig1
        
    def plot_single_image_with_circle_reticle(self, frame_nb=500, circle_diam = 250):
        eye_tracking_image = self.get_image(frame_nb = frame_nb)

        fig1=plt.figure()

        # we clip the range of the image for visualization
        bottom_scale=np.percentile(eye_tracking_image,5)            
        top_scale=np.percentile(eye_tracking_image,95)
        eye_tracking_image = np.clip(eye_tracking_image, bottom_scale, top_scale)
        eye_tracking_image = 255.0*(eye_tracking_image.astype('float')-bottom_scale)/(top_scale-bottom_scale)
        
        local_img=Image.fromarray(eye_tracking_image.astype('uint8'))
        local_img_rgb = Image.merge('RGB', (local_img,local_img,local_img)) 
        
        draw = ImageDraw.Draw(local_img_rgb)
        draw.line((0, 0) + local_img_rgb.size, fill=128)
        draw.line((0, local_img_rgb.size[1], local_img_rgb.size[0], 0), fill=128)
        draw.ellipse((local_img_rgb.size[0]/2-circle_diam/2, local_img_rgb.size[1]/2-circle_diam/2, local_img_rgb.size[0]/2+circle_diam/2, local_img_rgb.size[1]/2+circle_diam/2), fill = None, outline ='red')
        plt.imshow(local_img_rgb)

        
        
        return fig1
        
    def plot_var_image(self, temporal_subsampling = 100):
        sampled_frames = range(0, self.get_nb_frames(), temporal_subsampling)     
        nb_samples = len(sampled_frames)
        
        local_image = self.get_image(frame_nb = 1)

        # We init to float to ensure we have the precision to accumulate        
        var_image = np.zeros(local_image.shape,dtype=float)
        mean_image = np.zeros(local_image.shape,dtype=float)
        
        # We use an irerative formula of the var to avoid double passing
        for index in sampled_frames:
            local_image = self.get_image(frame_nb = index)
            var_image = var_image + local_image**2/(nb_samples - 1)
            mean_image = mean_image + local_image/nb_samples
            
        # n-1 as we have only an estimate of the mean
        var_image = var_image - nb_samples/(nb_samples-1)*mean_image**2     
        
        # We mormalize by mean value to equally look at all pixels
        var_image = var_image / mean_image     
            
        fig1=plt.figure()
        # We displayed using 5-95 qunatile limits
        bottom_scale=np.percentile(var_image,2)            
        top_scale=np.percentile(var_image,98)
        plt.imshow(var_image, cmap = 'gray', clim = [bottom_scale, top_scale])
        
        
        return fig1

        
    def plot_mean_image(self, temporal_subsampling = 100):
        sampled_frames = range(0, self.get_nb_frames(), temporal_subsampling)     
        nb_samples = len(sampled_frames)
        
        local_image = self.get_image(frame_nb = 1)

        # We init to float to ensure we have the precision to accumulate        
        mean_image = np.zeros(local_image.shape,dtype=float)
        
        # We use an irerative formula of the var to avoid double passing
        for index in sampled_frames:
            mean_image = mean_image + local_image/nb_samples
            
            
        fig1=plt.figure()
        # We displayed using 5-95 qunatile limits
        bottom_scale=np.percentile(mean_image,2)            
        top_scale=np.percentile(mean_image,98)
        plt.imshow(mean_image, cmap = 'gray', clim = [bottom_scale, top_scale])
        
        
        return fig1



