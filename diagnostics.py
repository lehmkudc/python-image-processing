# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 11:57:36 2018

@author: Dustin
"""

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

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
    
    for i in range( 0, round(x-2)):
        for j in range( 0, round(y-2)):
            r1 = round(i+3)
            r2 = round(j+3)
    
            Gx[i,j] = np.multiply( kx, img[i:r1,j:r2]).sum()+.01
            Gy[i,j] = np.multiply( ky, img[i:r1,j:r2]).sum()+.01
            GG[i,j] = math.sqrt(Gx[i,j]**2 + Gy[i,j]**2)
            Theta[i,j] = np.arctan(Gy[i,j]/Gx[i,j])*57.2958 # Degrees
        
    return list( (Gx, Gy, GG, Theta) )


def SFR_mask(img, reverse=False):
    # Apply SFR kernel to image
    k1 = np.array([1,1,0,0,-1,-1])
    k2 = np.array([1,1,1,1,0,0,-1,-1,-1,-1]) 
    
    k_SFR = 2*np.array([k2,k2,k2,k2,k2,-k2,-k2,-k2,-k2,-k2])
    if reverse ==True:
        k_SFR = k_SFR[::-1]
    img_SFR = apply_kernel( img, k_SFR)
    return img_SFR
    
def SFR_locate(img_SFR, threshold):
    # Use SFR mask to find the most likely SFR locations
    
    co_x = list(np.where(img_SFR >= threshold)[0].tolist())
    co_y = list(np.where(img_SFR >= threshold)[1].tolist())
    loc_x = []
    loc_y = []
    for i in range(0,len(co_x)):
        tgt = img_SFR[ co_x[i], co_y[i] ]
        if tgt < img_SFR[ co_x[i] +1, co_y[i] ]:
            continue
        if tgt < img_SFR[ co_x[i] -1, co_y[i] ]:
            continue
        if tgt < img_SFR[ co_x[i], co_y[i]+1 ]:
            continue
        if tgt < img_SFR[ co_x[i], co_y[i]-1 ]:
            continue
        if tgt < img_SFR[ co_x[i]+1, co_y[i]+1 ]:
            continue
        if tgt < img_SFR[ co_x[i]-1, co_y[i]-1 ]:
            continue
        if tgt < img_SFR[ co_x[i]-1, co_y[i]+1 ]:
            continue
        if tgt < img_SFR[ co_x[i]+1, co_y[i]-1 ]:
            continue
        loc_x.append( co_x[i] )
        loc_y.append( co_y[i] )
    loc_x = np.array( loc_x )
    loc_y = np.array( loc_y )
    
    plt.imshow(img_SFR,'gray')
    plt.plot(loc_y,loc_x,'ro', markersize=7, markeredgewidth=1, markerfacecolor='None')
    for i in range(0, len(loc_x)):
        coord = 'x:' + str(loc_x[i]) + '\ny:' + str(loc_y[i]) + '\ni:' + str(img_SFR[loc_x[i],loc_y[i]])
        plt.text( loc_y[i], loc_x[i], coord)
    
    return loc_x, loc_y


def SFR_zoom( img_SFR, x, y):
    # plot intensity of SFR mask at corner locations
    plt.figure('y axis')
    plt.xlim( y-10, y+10 )
    plt.xticks( np.linspace(y-10,y+10,11), rotation=90)
    for i in range(x-2,x+3):
        label = 'x:', +i 
        plt.plot(np.linspace(y-10,y+9,20,endpoint=True),img_SFR[i,(y-10):(y+10)], label=label)
    plt.legend()
    plt.figure('x axis')
    plt.xlim( x-10, x+10 )
    plt.xticks( np.linspace(x-10,x+10,11), rotation=90)
    for j in range(y-2,y+3):
        label = 'y:', + j 
        plt.plot(np.linspace(x-10,x+9,20,endpoint=True),img_SFR[(x-10):(x+10),j], label=label)
    plt.legend()

source = cv2.imread('Pictures for Dustin/RCCB_Quad_SFR.pgm',0)
source = rescale(source)
plt.imshow(source, 'gray')
k_gauss = 1/159*np.array([[2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]])
img_gauss = apply_kernel( source, k_gauss )
plt.imshow(img_gauss,'gray')
img_SFR = SFR_mask(img_gauss, False)
plt.imshow(img_SFR,'gray')
loc_x, loc_y = SFR_locate(img_SFR, 150)

SFR_zoom( img_SFR, loc_x[0], loc_y[0])


k2 = np.array([1,1,1,0,0,-1,-1,-1])
k = np.array([k2,k2,k2,k2,-k2,-k2,-k2,-k2])
k
