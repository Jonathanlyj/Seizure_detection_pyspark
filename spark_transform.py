#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 17:11:50 2019

@author: liyoujia
"""


import numpy as np
from scipy import signal
from scipy.signal import resample, hann
from sklearn import preprocessing
import os
#from spark_data_io import *









def upper_right_triangle(matrix):
    accum = []
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            accum.append(matrix[i, j])
    return np.array(accum)


def takeLog(data):
    
    indices = np.where(data <= 0)
    data[indices] = np.max(data)
    data[indices] = (np.min(data) * 0.1)
    return np.log10(data)
    

class FreqTransform:
    """
    Correlation in the frequency domain. First take FFT with (start, end) slice options,
    then calculate correlation co-efficients on the FFT output, followed by calculating
    eigenvalues on the correlation co-efficients matrix.

    The output features are (fft, upper_right_diagonal(correlation_coefficients), eigenvalues)

    Features can be selected/omitted using the constructor arguments.
    """
    def __init__(self, start, end, with_fft=False, with_corr=True, with_eigen=True):
        self.start = start
        self.end = end
        self.with_fft = with_fft
        self.with_corr = with_corr
        self.with_eigen = with_eigen
        assert with_corr or with_eigen
        assert end > start >= 0
        
    def get_name(self):
        selections = []
        if not self.with_corr:
            selections.append('nocorr')
        if not self.with_eigen:
            selections.append('noeig')
        if len(selections) > 0:
            selection_str = '-' + '-'.join(selections)
        else:
            selection_str = ''
        return 'freq-correlation-%d-%d-%s-%s' % (self.start, self.end, 'withfft' if self.with_fft else 'nofft',
                                                   selection_str)

    def apply(self, data):
        
        #FFT transform
        axis = data.ndim - 1
        data1 = np.fft.rfft(data, axis=axis)
        #Bandpass filter
        s = [slice(None),] * data.ndim
        s[-1] = slice(self.start, self.end)
        data1 = data1[s]
        #Get the log10 magnitude of each freq
        data1 = np.absolute(data1)
        data1 = takeLog(data1)
        #Scaling
        
        data2 = preprocessing.scale(data1, axis=0)
        
        #Get correlation matrix
        corr_data2 = np.corrcoef(data2)


        if self.with_eigen:
            
            w, v = np.linalg.eig(corr_data2)
            eigen_corr_data2  = np.absolute(w)
            eigen_corr_data2.sort()
        
        #Construct output frequency vector
        out = []
        if self.with_corr:
            corr_data2_flat = upper_right_triangle(corr_data2)
            out.append(corr_data2_flat)
        if self.with_eigen:
            out.append(eigen_corr_data2)
        if self.with_fft:
            data1 = data1.ravel()
            out.append(data1)
        for d in out:
           

            assert d.ndim == 1

        return np.concatenate(out, axis=0)
    
    
    
class TimeTransform:
    
    def __init__(self, max_hz, with_corr=True, with_eigen=True):
        self.max_hz = max_hz
        self.with_corr = with_corr
        self.with_eigen = with_eigen
        assert with_corr or with_eigen

    def get_name(self):
        selections = []
        if not self.with_corr:
            selections.append('nocorr')
        if not self.with_eigen:
            selections.append('noeig')
        if len(selections) > 0:
            selection_str = '-' + '-'.join(selections)
        else:
            selection_str = ''
        return 'time-correlation-r%d-%s' % (self.max_hz, selection_str)

    def apply(self, data):
        # so that correlation matrix calculation doesn't crash
        for ch in data:
            
            if np.alltrue(ch == 0.0):
                ch[-1] += 0.00001
        
        #Resample
        axis = data.ndim - 1
        if data.shape[-1] > self.max_hz:
            data1 = resample(data, self.max_hz, axis = axis)
        else:
            data1 = data
            
        #Scaling
        data1 = preprocessing.scale(data1, axis = 0)
        
        #Get correlation matrix
        corr_data1 = np.corrcoef(data1)
        
        if self.with_eigen:
            
            w, v = np.linalg.eig(corr_data1)
            eigen_corr_data1  = np.absolute(w)
            eigen_corr_data1.sort()

        out = []
        if self.with_corr:
            corr_data1_flat = upper_right_triangle(corr_data1)
            out.append(corr_data1_flat)
        if self.with_eigen:
            out.append(eigen_corr_data1)

        for d in out:
            assert d.ndim == 1

        return np.concatenate(out, axis=0)


class FreqWithTimeTransform:
    """
    Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
    """
    def __init__(self, start, end, max_hz):
        self.start = start
        self.end = end
        self.max_hz = max_hz


    def get_name(self):
        return 'fft-with-time-freq-corr-%d-%d-r%d-%s' % (self.start, self.end, self.max_hz)

    def apply(self, data):
        data1 = TimeTransform(self.max_hz).apply(data)
        data2 = FreqTransform(self.start, self.end, with_fft=True).apply(data)
        assert data1.ndim == data2.ndim
        return np.concatenate((data1, data2), axis=data1.ndim - 1)
    






    
    
        
    

