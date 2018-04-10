# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:44:58 2018

@author: Dustin
"""

import cv2
import numpy as np


cap = cv2.VideoCapture(0)
#frame = cv2.imread('Kumena Tyrant of Orazca.jpg',0)

while True:
    _, frame = cap.read()
    
    laplacian = cv2.Laplacian(frame, cv2.CV_64F)
    sobelx = cv2.Sobel( frame, cv2.CV_64F, 1,0,ksize=5)
    sobely = cv2.Sobel( frame, cv2.CV_64F, 0,1,ksize=5)
    canny = cv2.Canny( frame, 100, 100)
    
    cv2.imshow( 'original', frame)
    cv2.imshow('laplacian',laplacian)
    cv2.imshow('sobelx',sobelx)
    cv2.imshow('sobely',sobely)
    cv2.imshow('canny',canny)
    
    
    
    
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
