# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:49:10 2015

@author: jeromel
"""
import pandas as pd
import psycopg2
import os
import matplotlib.pyplot as plt
import numpy as np
# uncommented "import ophyse" due to path errors
# import ophyse

class LimsDatabase:
    def __init__(self, lims_id):            
        
        self.lims_id=lims_id
        
        # We first gather all information from LIMS       
        try:
            conn=psycopg2.connect(dbname="lims2", user="limsreader", host="limsdb2",password="limsro", port=5432)
            cur=conn.cursor()
            
            query=' '.join((
                "SELECT eso.id, eso.name, eso.storage_directory, eso.specimen_id"
                ", sp.external_specimen_name, eso.date_of_acquisition, u.login as operator",
                ", e.name as rig, id.depth, st.acronym, eso.parent_id, eso.workflow_state", 
                "FROM experiment_sessions eso JOIN specimens sp", 
                "ON sp.id=eso.specimen_id",
                "LEFT JOIN imaging_depths id ON id.id=eso.imaging_depth_id",
                "LEFT JOIN equipment e ON e.id=eso.equipment_id",
                "LEFT JOIN users u ON u.id=eso.operator_id",
                "LEFT JOIN structures st ON st.id=eso.targeted_structure_id",
                "WHERE eso.type='OphysExperiment' AND eso.id=%s",        
            ))

            cur.execute(query,[lims_id])
            
            lims_data = cur.fetchall()
            if lims_data == []:
                self.data_present = False
            else:
                self.data_pointer = lims_data[0]   
                self.data_present = True
                
            conn.close()  
        except:
            print("Unable to query LIMS database")
            self.data_present = False

    def is_valid(self):
        return self.data_present
        
    def get_qc_param(self):
        if not(hasattr(self,'qc_data')):
            qc_data = pd.DataFrame()
            
            qc_data['lims_id']=[self.get_lims_id()]
            qc_data['datafolder']=[self.get_datafolder()]
            qc_data['specimen_id']=[self.get_specimen_id()]
            qc_data['external_specimen_id']=[self.get_external_specimen_id()]
            qc_data['experiment_name']=[self.get_experiment_name()]
            qc_data['experiment_date']=[self.get_experiment_date()]
            qc_data['rig']=[self.get_rig()]
            qc_data['depth']=[self.get_depth()]
            qc_data['operator']=[self.get_operator()]
            qc_data['structure']=[self.get_structure()]
            qc_data['parent']=[self.get_parent()]
            qc_data['workflow_state']=[self.get_workflow_state()]
            qc_data['specimen_driver_line']=[self.get_specimen_driver_line()]

            # We save the qc internally
            self.qc_data=qc_data
            
        return self.qc_data        
                 
    def get_all_ophys_lims_columns_names(self):
            conn=psycopg2.connect(dbname="lims2", user="limsreader", host="limsdb2",password="limsro", port=5432)
            cur=conn.cursor()
            
            query=' '.join((
                "SELECT column_name "
                "FROM information_schema.columns",
                "WHERE table_name   = 'experiment_sessions'",        
            ))

            cur.execute(query)
            
            lims_data = cur.fetchall()
            conn.close()  
            return lims_data

    def save_qc_param(self,saved_folder):
        self.get_qc_param()        
        file_qc=os.path.join(saved_folder,'lims_database_qcdata.pkl')
        self.qc_data.to_pickle(file_qc)             

    def plot_qc(self):
        if self.data_present:
            try:
                self.plot_sync_report()
                self.plot_vasculature()
                self.plot_two_photon_surf()
                self.plot_two_photon_depth()
            except:
                print("Error generating plots")
                
    def save_plot_qc(self,saved_folder):
        if self.data_present:
            try:
                sync_report_png=os.path.join(saved_folder,'sync_report.png')
                fig=self.plot_sync_report()
                fig.savefig(sync_report_png)
                plt.close(fig)

                vasculature_png=os.path.join(saved_folder,'vasculature.png')
                fig=self.plot_vasculature()
                fig.savefig(vasculature_png)
                plt.close(fig)

                two_photon_surf_png=os.path.join(saved_folder,'two_photon_surf.png')
                fig=self.plot_two_photon_surf()
                fig.savefig(two_photon_surf_png)
                plt.close(fig)

                two_photon_depth_png=os.path.join(saved_folder,'two_photon_depth.png')
                fig=self.plot_two_photon_depth()
                fig.savefig(two_photon_depth_png)
                plt.close(fig)
                
                isi_target_png=os.path.join(saved_folder,'isi_target.png')
                fig=self.plot_isi_target_image()
                fig.savefig(isi_target_png)
                plt.close(fig)
                
                local_parent = self.get_parent_ophyse()
                
                if local_parent.data_present:
                    parent_vasculature_png=os.path.join(saved_folder,'parent_vasculature.png')
                    fig=local_parent.plot_vasculature()
                    fig.savefig(parent_vasculature_png)
                    plt.close(fig)
    
                    parent_two_photon_surf_png=os.path.join(saved_folder,'parent_two_photon_surf.png')
                    fig=local_parent.plot_two_photon_surf()
                    fig.savefig(parent_two_photon_surf_png)
                    plt.close(fig)
    
                    parent_two_photon_depth_png=os.path.join(saved_folder,'parent_two_photon_depth.png')
                    fig=local_parent.plot_two_photon_depth()
                    fig.savefig(parent_two_photon_depth_png)
                    plt.close(fig)
            except:
                print("Error generating plots")
                
    def get_datafolder(self):
        if not(hasattr(self,'data_folder')):
            data_folder = self.data_pointer[2]
            
            # We need to convert internal storage path to real path on titan
            data_folder=data_folder.replace('/','\\')
            data_folder=data_folder.replace('\\projects','\\\\titan\\cns')
            data_folder=data_folder.replace('\\vol1','')  
            self.data_folder=data_folder     
        
        return self.data_folder           

    def get_parent_ophyse(self):
        
        lims_id = self.get_parent()
        parent_ophys = ophyse.data_flow.LimsDatabase(str(lims_id))

        return parent_ophys   
        
    def get_isi_datafolder(self):
        isi_folder = self.get_isi()
        isi_folder = isi_folder[1]
                
        # We need to convert internal storage path to real path on titan
        isi_folder=isi_folder.replace('/','\\')
        isi_folder=isi_folder.replace('\\projects','\\\\titan\\cns')
        isi_folder=isi_folder.replace('\\vol1','')  
        
        return isi_folder           

    def get_isi(self):
        lims_id = self.get_parent()
        isi_found = False
        
        conn=psycopg2.connect(dbname="lims2", user="limsreader", host="limsdb2",password="limsro", port=5432)
        cur=conn.cursor()
        
        iteration = 0
        
        while not(isi_found):

            query=' '.join((
                "SELECT eso.type, eso.storage_directory, eso.parent_id", 
                "FROM experiment_sessions eso", 
                "WHERE eso.id=%s",        
            ))
                
            cur.execute(query,[lims_id])
            
            lims_data = cur.fetchall()
            lims_data = lims_data[0]
            lims_type = lims_data[0]   
                        
            if lims_type == 'IsiExperiment':
                final_lims = lims_id
                lims_folder = lims_data[1]
                isi_found = True
            elif lims_type == 'OphysExperiment':
                isi_found = False
                lims_id = lims_data[2]
            else:
                isi_found = True
                final_lims = []
                lims_folder = []
            
            iteration = iteration + 1
            
            # To prevent infinite loops in case ophyse experiments are cross-referencing            
            if iteration >10:
                isi_found = True
                final_lims = []
                lims_folder = []
                
        conn.close() 
        
        return final_lims, lims_folder
        
    def get_specimen_id(self):
        return self.data_pointer[3]
        
    def get_external_specimen_id(self):
        return self.data_pointer[4]

    def get_experiment_name(self):
        return self.data_pointer[1]

    def get_workflow_state(self):
        return self.data_pointer[11]
        
    def get_specimen_driver_line(self):
          
        try:
            conn=psycopg2.connect(dbname="lims2", user="limsreader", host="limsdb2",password="limsro", port=5432)
            cur=conn.cursor()
            
            query=' '.join((
                "SELECT g.name as driver_line", 
                "FROM experiment_sessions eso JOIN specimens sp", 
                "ON sp.id=eso.specimen_id",
                "JOIN donors d ON d.id=sp.donor_id",
                "JOIN donors_genotypes dg ON dg.donor_id=d.id",
                "JOIN genotypes g ON g.id=dg.genotype_id",
                "JOIN genotype_types gt ON gt.id=g.genotype_type_id AND gt.name = 'driver'",
                "WHERE eso.type='OphysExperiment' AND eso.id=%s",              
            ))
    
            cur.execute(query,[self.lims_id])
            genotype_data = cur.fetchall()
                            
            final_genotype = ''        
            link_string = ''
            for local_text in genotype_data:
                local_gene=local_text[0]
                final_genotype=final_genotype+link_string+local_gene
                link_string = ';'    
          
            conn.close()  

        except:
            final_genotype = ''
        
        return final_genotype

    def get_specimen_reporter_line(self):
          
        try:
            conn=psycopg2.connect(dbname="lims2", user="limsreader", host="limsdb2",password="limsro", port=5432)
            cur=conn.cursor()
            
            query=' '.join((
                "SELECT g.name as reporter_line", 
                "FROM experiment_sessions eso JOIN specimens sp", 
                "ON sp.id=eso.specimen_id",
                "JOIN donors d ON d.id=sp.donor_id",
                "JOIN donors_genotypes dg ON dg.donor_id=d.id",
                "JOIN genotypes g ON g.id=dg.genotype_id",
                "JOIN genotype_types gt ON gt.id=g.genotype_type_id AND gt.name = 'reporter'",
                "WHERE eso.type='OphysExperiment' AND eso.id=%s",              
            ))
    
            cur.execute(query,[self.lims_id])
            genotype_data = cur.fetchall()
                            
            final_genotype = ''        
            link_string = ''
            for local_text in genotype_data:
                local_gene=local_text[0]
                final_genotype=final_genotype+link_string+local_gene
                link_string = ';'    
          
            conn.close()  

        except:
            final_genotype = ''
        
        return final_genotype

    def get_lims_id(self):
        return self.lims_id

    def get_experiment_date(self):
        from pytz import timezone
        
        utc_time = self.data_pointer[5]
        zoned_time = utc_time.replace(tzinfo=timezone('UTC'))
        correct_time = zoned_time.astimezone(timezone('US/Pacific'))
        return correct_time.replace(tzinfo=None)
        
    def get_operator(self):
        return self.data_pointer[6]
        
    def get_rig(self):
        return self.data_pointer[7]
        
    def get_depth(self):
        return self.data_pointer[8]    

    def get_structure(self):
        return self.data_pointer[9]    

    def get_parent(self):
        return self.data_pointer[10]            
        
    def plot_isi_target_image(self):
        exp_folder = self.get_isi_datafolder()
        fig1=plt.figure()
        
        if not(exp_folder == []):   
            for file in os.listdir(exp_folder):
                if file.endswith("_target_map.tif"):  
                    full_path = os.path.join(exp_folder, file)
                    local_im = plt.imread(full_path)
                    plt.imshow(local_im, cmap = 'gray')
        return fig1
        
    def plot_two_photon_depth(self):
        exp_folder = self.get_datafolder()
        fig1=plt.figure()

        for file in os.listdir(exp_folder):
            if file.endswith("_averaged_depth.tif"):  
                full_path = os.path.join(exp_folder, file)
                local_im = plt.imread(full_path)
                bottom_scale=np.percentile(local_im,1)            
                top_scale=np.percentile(local_im,99)
                plt.imshow(local_im, cmap = 'gray', clim = [bottom_scale, top_scale])
        return fig1
        
    def plot_two_photon_surf(self):
        exp_folder = self.get_datafolder()
        fig1=plt.figure()

        for file in os.listdir(exp_folder):
            if file.endswith("_averaged_surface.tif"):  
                full_path = os.path.join(exp_folder, file)
                local_im = plt.imread(full_path)
                bottom_scale=np.percentile(local_im,1)            
                top_scale=np.percentile(local_im,99)
                plt.imshow(local_im, cmap = 'gray', clim = [bottom_scale, top_scale])
        return fig1
        
    def plot_vasculature(self):
        exp_folder = self.get_datafolder()
        fig1=plt.figure()

        for file in os.listdir(exp_folder):
            if file.endswith("_vasculature.tif"):    
                full_path = os.path.join(exp_folder, file)
                local_im = plt.imread(full_path)
                plt.imshow(local_im, cmap = 'gray')        
        return fig1
        
    def plot_sync_report(self):
        exp_folder = self.get_datafolder()
        fig1=plt.figure()

        for file in os.listdir(exp_folder):
            if file.endswith("_sync_report.png"):            
                full_path = os.path.join(exp_folder, file)
                local_im = plt.imread(full_path)
                plt.axis('off')
                plt.imshow(local_im)        
        return fig1

    def get_video_directory(self):
        return self.get_datafolder()