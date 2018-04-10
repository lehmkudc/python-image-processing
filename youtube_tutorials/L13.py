# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 08:47:31 2018

@author: Dustin
"""

import cv2
import numpy as np

img = cv2.imread('opencv-corner-detection-sample.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray, 200, 0.01, 10)
corners = np.int0(corners)

for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img, (x,y), 3, 255, -1)

cv2.imshow('corner',img)