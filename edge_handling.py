# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 09:26:50 2018

@author: Dustin
"""
import numpy as np
import cv2
import sfr_detection_functions as sfr
import sys
import matplotlib.pyplot as plt

img = cv2.imread('sfr_kern.png',0)
n = 3

    # Creates a ring of 0 values around an image of thickness n
x,y = img.shape
b = np.zeros( (x+2*n,y+2*n) )
b[n:(x+n),n:(y+n)] = img


b[0:n,0:n] = img[0,0]
b[(x+n):(x+2*n), (y+n):(y+2*n)] = img[x-1,y-1]
b[(x+n):(x+2*n), 0:n] = img[x-1,0]
b[0:n, (y+n):(y+2*n)] = img[0,y-1]

b[0:n,n:(y+n)] = img[0,:]
b[n:(x+n),0:n] = img[:,0:1]
b[(x+n):(x+2*n),n:(y+n)] = img[(x-1),:]
b[n:(x+n),(y+n):(y+2*n)] = img[:,(y-1):y]


plt.imshow(b,'gray')
