#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:46:15 2019

@author: liyoujia
"""




# coding: utf-8

# In[1]:

from scipy.io import loadmat
import glob
import os
import gcsfs 


# In[21]:

class dataloader:
    
    def __init__(self, folder_path, gcp_fs):
        self.path = folder_path
        self.data = []
        self.fs = gcp_fs
        
    def load_ictal_data(self):
        self.data = []

        for filename in self.fs.glob(os.path.join(self.path, '*_ictal_segment*')):
            file = fs.open(filename, 'rb')
            data = loadmat(file)
            self.data.append(data)
            file.close()
        return self.data
            
        
    def load_interictal_data(self):
        self.data = []

        for filename in self.fs.glob(os.path.join(self.path, '*_interictal_segment*')):
            file = fs.open(filename, 'rb')
            data = loadmat(file)
            self.data.append(data)
            file.close()
        return self.data

    def load_test_data(self):
        self.data = []
        self.names = []

        for filename in self.fs.glob(os.path.join(self.path, '*test_segment*')):
            file = fs.open(filename, 'rb')
            data = loadmat(file)
            self.data.append(data)
            sample_name = filename.split(".")[0].split("/")[-1]
            self.names.append(sample_name)
            file.close()
        return self.data, self.names

        
        




#if __name__ == '__main__':
#    loader = dataloader('seizure-data/Patient_8')
#    ictal_samples = loader.load_ictal_data()

#    print(ictal_samples[0]['data'])









