import numpy as np
import cv2
import sfr_detection_functions as sfr
import matplotlib.pyplot as plt
from scipy import ndimage, signal
import math
import argparse

cap = cv2.VideoCapture(0)
k2 = np.array([1,1,1,1,0,-1,-1,-1,-1]) 
k_SFR = np.array([k2,k2,k2,k2,np.zeros((9,)),-k2,-k2,-k2,-k2])

while True:
    ret, img = cap.read()

    
    #img_SFR = sfr.ring( img, 4 )
    img_SFR = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_SFR = signal.fftconvolve( img_SFR, k_SFR,'full' )
    img_SFR = sfr.rescale(img_SFR)
    
    img = sfr.ring( img, 4)
    img = sfr.rescale(img)
    
    iSFR = (255 -img_SFR)
    iSFR = sfr.rescale(iSFR)
    
    loc_x, loc_y = sfr.SFR_locate(img_SFR, 200)
    sfr.SFR_apply_cv2( img_SFR, loc_x, loc_y,name='SFRmask'  )
    sfr.SFR_apply_cv2( img, loc_x, loc_y, sfr_mask = img_SFR,name='original')
    
    
    iloc_x, iloc_y = sfr.SFR_locate(iSFR, 200)
    sfr.SFR_apply_cv2( iSFR, iloc_x, iloc_y,color=(0,255,0) ,name='SFRmask' )
    sfr.SFR_apply_cv2( img, iloc_x, iloc_y, sfr_mask = iSFR,name='original',color=(0,255,0))

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


sfr.loc_inrange(loc_x,loc_y,(252,383),(313,441))
sfr.loc_inrange(iloc_x,iloc_y,(252,383),(313,441))

#def set_checkerboard_range()


print(loc_x, loc_y)
print(iloc_x,iloc_y)