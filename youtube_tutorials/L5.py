# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:42:18 2018

@author: Dustin
"""

import cv2
import numpy as np

img = cv2.imread('C:/Users/Dustin/Desktop/photo_2018-02-13_21-03-40.jpg', cv2.IMREAD_COLOR)

im1 = cv2.imread('C:/Users/Dustin/Desktop/Kumena Tyrant of Orazca.jpg', cv2.IMREAD_COLOR)
im2 = cv2.imread('C:/Users/Dustin/Desktop/Sidon Zora Prince.jpg', cv2.IMREAD_COLOR)

#add = im1 + im2
add = cv2.add(im1,im2)
#cv2.imshow('add',add)

weighted = cv2.addWeighted(im1, 0.6, im2, 0.4, 0)
#cv2.imshow('weighted',weighted)

rows, cols, channels = im1.shape

roi = img[0:rows, 0:cols]

img2gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',img2gray)
ret, mask = cv2.threshold(img2gray, 60, 255, cv2.THRESH_BINARY)
cv2.imshow('mask', mask)
mask_inv = cv2.bitwise_not(mask)
cv2.imshow('notmask',mask_inv)
im1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
cv2.imshow('bg',im1_bg)
img2_fg = cv2.bitwise_and(im1,im1,mask=mask)
cv2.imshow('fg',img2_fg)
dst = cv2.add(im1_bg, img2_fg)
cv2.imshow('dst',dst)
img[0:rows, 0:cols] = dst

cv2.imshow('image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()