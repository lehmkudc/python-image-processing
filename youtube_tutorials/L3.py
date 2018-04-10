# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 09:19:20 2018

@author: Dustin
"""

import numpy as np
import cv2

img = cv2.imread('C:/Users/Dustin/Desktop/photo_2018-02-13_21-03-40.jpg', cv2.IMREAD_COLOR)

cv2.line( img, (0,0), (150,150), (255,0,0), 15)

cv2.rectangle( img, (15,25), (200,150),(0,255,0), 5)

cv2
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
