# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:21:45 2018

@author: Dustin
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import sys


def ring(im, n):
    # Creates a ring of 0 values around an image of thickness n
    x,y = im.shape
    b = np.zeros( (x+2*n,y+2*n) )
    b[n:(x+n),n:(y+n)] = im
    return b

def rescale(img):
    # Sets the scale of an image into integers from 0 to 255
    imin = img.min()
    img = img - imin
    imax = img.max()
    img = ((img/imax)*255).astype('uint8')
    return img

def apply_kernel(img, kernel):
    # Creates a mask by applying a kernal to an image img
    n = kernel.shape[0] - 1
    x,y = img.shape
    mask = np.ones( (x-n,y-n) )
    
    for i in range( 0, round(x-n)):
        for j in range( 0, round(y-n)):
            r1 = round(i+n+1)
            r2 = round(j+n+1)
            
            mask[i,j] = np.multiply( kernel, img[i:r1,j:r2] ).sum()
    
    mask = rescale(mask)
    
    return mask

def find_gradient(img):
    # Determine x, y, and total gradient as well as angle in deg
    x,y = img.shape
    kx = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    ky = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    Gx = np.ones((x-2,y-2))
    Gy = np.ones((x-2,y-2))
    GG = np.ones((x-2,y-2))
    Theta = np.ones((x-2,y-2))
    
    for i in range( 0	, round(x-2)):
        for j in range( 0, round(y-2)):
            r1 = round(i+3)
            r2 = round(j+3)
    
            Gx[i,j] = np.multiply( kx, img[i:r1,j:r2]).sum()+.01
            Gy[i,j] = np.multiply( ky, img[i:r1,j:r2]).sum()+.01
            GG[i,j] = math.sqrt(Gx[i,j]**2 + Gy[i,j]**2)
            Theta[i,j] = np.arctan(Gy[i,j]/Gx[i,j])*57.2958 # Degrees
        
    return list( (Gx, Gy, GG, Theta) )

source = cv2.imread('SFRreg_pillared_tilted_trans.png',0)
cv2.imshow('source (grayscale)', source)

# Apply Gaussian Blur
k_gauss = 1/159*np.array([[2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]])
img_gauss = apply_kernel( source, k_gauss )
cv2.imshow('gaussian blur', img_gauss)

# Search for Provided corner criteria
k_corner = np.array([[1,1,1,-1,-1,-1],[1,1,1,-1,-1,-1],[1,1,1,-1,-1,-1],
                     [-1,-1,-1,1,1,1],[-1,-1,-1,1,1,1],[-1,-1,-1,1,1,1]])
img_corner = apply_kernel( img_gauss, k_corner)
cv2.imshow('corner search',img_corner)
x = list(set(np.where(img_corner >= 200)[0].tolist()))
y = list(set(np.where(img_corner >= 200)[1].tolist()))

plt.figure('At x axis pixels')
plt.xlabel('yloc')
for i in range( 0, len(x)):
    plt.plot(img_corner[x[i],:])


plt.figure('At y axis pixels')
plt.xlabel('xloc')
for j in range( 0, len(y)):
    plt.plot(img_corner[:,y[j]])


Gx, Gy, GG, Theta = find_gradient(img_gauss)

plt.imshow(GG, 'gray')


sys.exit()



# Non-Max suppression
deg = np.ones( GG.shape )
Gn = np.copy(GG)
for i in range( 1, GG.shape[0]-1):
    for j in range( 1, GG.shape[1]-1):
        deg[i,j] = round(Theta[i,j]/45)*45
        if (deg[1,j] == 1):
            Gn[i,j] = 1
        elif (deg[1,j] == 0):
            if ( (GG[i,j] < GG[i-1,j]) | (GG[i,j] < GG[i+1,j])):
                Gn[i,j] = 1
        elif (deg[1,j] == 90):
            if ( (GG[i,j] < GG[i,j-1]) | (GG[i,j] < GG[i,j+1])):
                Gn[i,j] = 1
        elif (deg[1,j] == -90):
            if ( (GG[i,j] < GG[i,j-1]) | (GG[i,j] < GG[i,j+1])):
                Gn[i,j] = 1
        elif (deg[1,j] == 45):
            if ( (GG[i,j] < GG[i-1,j-1]) | (GG[i,j] < GG[i+1,j+1])):
                Gn[i,j] = 1
        elif (deg[1,j] == -45):
            if ( (GG[i,j] < GG[i-1,j+1]) | (GG[i,j] < GG[i+1,j-1])):
                Gn[i,j] = 1
        else:
            sys.exit("PICNIC")



# Thresholding
t1 = 50.
t2 = 150.
edgy = np.ones( GG.shape)
for i in range( 0, GG.shape[0]):
    for j in range( 0, GG.shape[1]):
        if ( Gn[i,j] >= t2 ):
            edgy[i,j] = 255
        elif ( (Gn[i,j] >= t1) & (Gn[i,j] <= t2)):
            edgy[i,j] = 155
        else:
            edgy[i,j] = 1

plt.imshow(edgy,'gray')
#plt.imshow(canny,'gray')
