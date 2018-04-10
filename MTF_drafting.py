# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 09:16:15 2018

@author: Dustin
"""


import cv2
import numpy as np
import sfr_detection_functions as sfr
import matplotlib.pyplot as plt

sfr_reference = cv2.imread('SFRreg_pillared_tilted_trans.png',0)
img = sfr_reference[ 120:340,190:410]

for i in range( 0,img.shape[0]):
    for j in range( 0,img.shape[1]):
        if img[i,j] < 119:
            img[i,j] = 119
        elif img[i,j] > 223:
            img[i,j] = 223
            



img_s = sfr.rescale( img)
plt.figure()
plt.plot( img_s[100,:])


g1 = sfr.gauss_blur(img_s)
plt.plot( g1[100,:])

g2 = sfr.gauss_blur(g1)
plt.plot( g2[100,:])

g3 = sfr.gauss_blur(g2)
plt.plot( g3[100,:])

def diff(x):
    y = np.zeros(len(x)-2)
    for i in range(1,len(x)-2):
        y[i] = -0.5*x[i-1] + 0.5*x[i+1]
    
    return y
     
dimg = diff(g3[100,:])

fimg = np.fft.rfft( dimg )
plt.figure()   
plt.plot( fimg)


