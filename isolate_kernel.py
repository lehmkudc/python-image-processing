# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 15:28:27 2018

@author: Dustin
"""
import cv2
import sfr_detection_functions as sfr
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal




sfr_reference = cv2.imread('Pictures for Dustin/RCCB_Quad_SFR.pgm',0)
sfr_reference = sfr.rescale( sfr_reference)

x,y = sfr_reference.shape

ham_thresh = np.zeros((x,y))
thresh = 70
for i in range(0,x):
    for j in range(0,y):
        if sfr_reference[i,j] > thresh:
            ham_thresh[i,j] = 255
        else:
            ham_thresh[i,j] = 0

#plt.imshow(ham_thresh,'gray')


#sfr.SFR_routine_cv2(sfr_reference)
'''
k2 = np.array([1,1,1,1,0,-1,-1,-1,-1]) 
k_SFR = np.array([k2,k2,k2,k2,np.zeros((9,)),-k2,-k2,-k2,-k2])
img_SFR = ndimage.convolve( sfr_reference, k_SFR, )
img_SFR = sfr.rescale(img_SFR)
img_SFR1 = signal.fftconvolve(sfr_reference, k_SFR,'full')
img_SFR1 = sfr.rescale(img_SFR1)
img_SFR2 = sfr.apply_kernel(sfr_reference, k_SFR)



cv2.imshow('prebuilt',img_SFR)
cv2.imshow('fft',img_SFR1)
cv2.imshow('my algorithm', img_SFR2)
'''

sfr.SFR_routine( ham_thresh, 160 )