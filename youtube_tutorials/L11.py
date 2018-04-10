# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:44:58 2018

@author: Dustin
"""

import cv2
import numpy as np


source = cv2.imread('photo_2018-02-26_08-21-14.jpg')
gray_source = cv2.cvtColor(source,cv2.COLOR_BGR2GRAY)


star = cv2.imread('Starbucks-logo.png',0)
target = cv2.resize(star, (0,0), fx=0.3,fy=0.3)
w,h = target.shape[::-1]

res = cv2.matchTemplate(gray_source, target, cv2.TM_CCOEFF_NORMED)
threshold = 0.5
loc = np.where(res >= threshold)
print(res)



cv2.imshow('source',gray_source)
cv2.imshow('target',target)
cv2.imshow('res',res)