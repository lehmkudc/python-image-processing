# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 07:38:23 2018

@author: Dustin
"""

'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('C:/Users/Dustin/Desktop/photo_2018-02-13_21-03-40.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('image', img)

plt.imshow(img, cmap='gray')
plt.plot( [80,20], [30,60], 'c', linewidth=5)
plt.show()
'''

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(frame)
    
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


cap.isOpened()


