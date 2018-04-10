# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:02:18 2018

@author: Dustin
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

source = cv2.imread('photo_2018-02-26_08-21-14.jpg')
source = cv2.cvtColor(source,cv2.COLOR_BGR2GRAY)
source = cv2.Canny( source, 100, 100)

star = cv2.imread('Starbucks-logo.png',0)
target = cv2.resize(star, (0,0), fx=0.3,fy=0.3)
target = cv2.Canny( target, 100, 100)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(source,None)
kp2, des2 = orb.detectAndCompute(target,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches( source, kp1, target, kp2, matches[:10], None, flags=2)
plt.imshow(img3)