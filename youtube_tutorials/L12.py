# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 08:35:29 2018

@author: Dustin
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('photo_2018-02-26_08-21-14.jpg')
mask = np.zeros(img.shape[:2], np.uint8)

bgdModel = np.zeros( (1,65), np.float64)
fgdModel = np.zeros( (1,65), np.float64)

rect = ( 200, 100, 200, 400 )

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask ==0), 0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.imshow(img)
plt.colorbar()
plt.show()