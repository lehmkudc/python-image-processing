# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 09:44:14 2018

@author: Dustin
"""

import numpy as np
import cv2

img = cv2.imread('C:/Users/Dustin/Desktop/photo_2018-02-13_21-03-40.jpg', cv2.IMREAD_COLOR)

px = img[55,55]

img[55,55] = [255,255,255]

img[100:150, 100:150] = [255,255,255]
cv2.imshow('image',img)


