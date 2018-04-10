import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from scipy import signal

def ring(im, n, edge='extend', asym=False):
    # Creates a ring of values around an image of thickness n
    
    if im.ndim == 2:
        x,y = im.shape
        b = np.zeros( (x+2*n,y+2*n) )
        if asym==True:
            b = np.zeros( (x+2*n + 1,y+2*n + 1) )
        b[n:(x+n),n:(y+n)] = im
        
        if edge =='extend':
            b[0:n,0:n] = im[0,0]
            b[(x+n):(x+2*n), (y+n):(y+2*n)] = im[x-1,y-1]
            b[(x+n):(x+2*n), 0:n] = im[x-1,0]
            b[0:n, (y+n):(y+2*n)] = im[0,y-1]
            
            b[0:n,n:(y+n)] = im[0,:]
            b[n:(x+n),0:n] = im[:,0:1]
            b[(x+n):(x+2*n),n:(y+n)] = im[(x-1),:]
            b[n:(x+n),(y+n):(y+2*n)] = im[:,(y-1):y]
        return b

    else:
        x,y,z = im.shape
        b = np.zeros( (x+2*n,y+2*n,z) )
        if asym==True:
            b = np.zeros( (x+2*n + 1,y+2*n + 1,z) )
        b[n:(x+n),n:(y+n),:] = im
        
        if edge =='extend':
            b[0:n,0:n,:] = im[0,0,:]
            b[(x+n):(x+2*n), (y+n):(y+2*n),:] = im[x-1,y-1,:]
            b[(x+n):(x+2*n), 0:n,:] = im[x-1,0,:]
            b[0:n, (y+n):(y+2*n),:] = im[0,y-1,:]
            
            b[0:n,n:(y+n),:] = im[0,:,:]
            b[n:(x+n),0:n,:] = im[:,0:1,:]
            b[(x+n):(x+2*n),n:(y+n),:] = im[(x-1),:,:]
            b[n:(x+n),(y+n):(y+2*n),:] = im[:,(y-1):y,:]
        return b


def rescale(img):
    # Sets the scale of an image into integers from 0 to 255
    imin = img.min()
    img = img - imin
    imax = img.max()
    img = ((img/imax)*255).astype('uint8')
    return img

def apply_kernel(img, kernel):
    # Creates a mask by applying a kernal to an image img
    n = kernel.shape[0] - 1
    x,y = img.shape
    mask = np.ones( (x-n,y-n) )
    
    for i in range( 0, round(x-n)):
        for j in range( 0, round(y-n)):
            r1 = round(i+n+1)
            r2 = round(j+n+1)
            
            mask[i,j] = np.multiply( kernel, img[i:r1,j:r2] ).sum()
    
    mask = rescale(mask)
    
    return mask

def gauss_blur(img):
    img = ring(img, 2)
    k_gauss = 1/159*np.array([[2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]])
    img_gauss = apply_kernel( img, k_gauss )
    return img_gauss

def find_gradient(img):
    # Determine x, y, and total gradient as well as angle in deg
    img = ring( img, 1)
    x,y = img.shape
    kx = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    ky = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    Gx = np.ones((x-2,y-2))
    Gy = np.ones((x-2,y-2))
    GG = np.ones((x-2,y-2))
    Theta = np.ones((x-2,y-2))
    
    for i in range( 0, round(x-2)):
        for j in range( 0, round(y-2)):
            r1 = round(i+3)
            r2 = round(j+3)
    
            Gx[i,j] = np.multiply( kx, img[i:r1,j:r2]).sum()+.01
            Gy[i,j] = np.multiply( ky, img[i:r1,j:r2]).sum()+.01
            GG[i,j] = math.sqrt(Gx[i,j]**2 + Gy[i,j]**2)
            Theta[i,j] = np.arctan(Gy[i,j]/Gx[i,j])*57.2958 # Degrees
        
    return list( (Gx, Gy, GG, Theta) )


def SFR_mask(img, reverse=False):
    # Apply SFR kernel to image
    img = ring(img, 4)
    k1 = np.array([1,1,1,1,-1,-1,-1,-1,-1]) 
    k2 = np.array([1,1,1,1,0,1,1,1,1])
    k3 = np.array([-1,-1,-1,-1,-1,1,1,1,1])
    
    k_SFR = np.array([k1,k1,k1,k1,k2,k3,k3,k3,k3])
    if reverse ==True:
        k_SFR = k_SFR[::-1]
    img_SFR = apply_kernel( img, k_SFR)
    return img_SFR

def SFR_maskfft(img, reverse=False):
    # Apply SFR kernel to image
    img = ring(img, 4)
    k1 = np.array([1,1,1,1,-1,-1,-1,-1,-1]) 
    k2 = np.array([1,1,1,1,0,1,1,1,1])
    k3 = np.array([-1,-1,-1,-1,-1,1,1,1,1])
    
    k_SFR = np.array([k1,k1,k1,k1,k2,k3,k3,k3,k3])
    if reverse ==True:
        k_SFR = k_SFR[::-1]
    img_SFR = signal.fftconvolve( img, k_SFR, 'full')
    img_SFR = rescale(img_SFR)
    return img_SFR


def SFR_locate(img_SFR, threshold):
    # Use SFR mask to find the most likely SFR locations
    
    co_x = list(np.where(img_SFR >= threshold)[0].tolist())
    co_y = list(np.where(img_SFR >= threshold)[1].tolist())
    loc_x = []
    loc_y = []
    
    for i in range( 0, len(co_x)):    
    
        tgt = img_SFR[ co_x[i], co_y[i] ]
        if ( (co_x[i] < 15) | (co_x[i] > (img_SFR.shape[0]-15) )):
            pass
        elif ( (co_y[i] < 15) | (co_y[i] > (img_SFR.shape[1]-15))):
            pass
        elif tgt < img_SFR[ co_x[i] +1, co_y[i] ]:
            pass
        elif tgt < img_SFR[ co_x[i] -1, co_y[i] ]:
            pass
        elif tgt < img_SFR[ co_x[i], co_y[i]+1 ]:
            pass
        elif tgt < img_SFR[ co_x[i], co_y[i]-1 ]:
            pass
        elif tgt < img_SFR[ co_x[i]+1, co_y[i]+1 ]:
            pass
        elif tgt < img_SFR[ co_x[i]-1, co_y[i]-1 ]:
            pass
        elif tgt < img_SFR[ co_x[i]-1, co_y[i]+1 ]:
            pass
        elif tgt < img_SFR[ co_x[i]+1, co_y[i]-1 ]:
            pass
        elif img_SFR[ (co_x[i]-1):(co_x[i]+1),(co_y[i]-1):(co_y[i]+1)].mean() < threshold:
            pass
        else:    
            loc_x.append( co_x[i] )
            loc_y.append( co_y[i] )
    loc_x = np.array( loc_x )
    loc_y = np.array( loc_y )
    
    return loc_x, loc_y


def SFR_zoom( img_SFR, x, y):
    # plot intensity of SFR mask at corner locations
    plt.figure('y axis')
    plt.xlim( y-10, y+10 )
    plt.xticks( np.linspace(y-10,y+10,11), rotation=90)
    for i in range(x-2,x+3):
        label = 'x:', +i 
        plt.plot(np.linspace(y-10,y+9,20,endpoint=True),img_SFR[i,(y-10):(y+10)], label=label)
    plt.legend()
    plt.figure('x axis')
    plt.xlim( x-10, x+10 )
    plt.xticks( np.linspace(x-10,x+10,11), rotation=90)
    for j in range(y-2,y+3):
        label = 'y:', + j 
        plt.plot(np.linspace(x-10,x+9,20,endpoint=True),img_SFR[(x-10):(x+10),j], label=label)
    plt.legend()
    
    
def SFR_apply(img, loc_x, loc_y, Title='Image', sfr_mask='default'):
    if sfr_mask == 'default':
        sfr_mask = img
    plt.figure(Title)
    plt.imshow(img,'gray')
    plt.plot(loc_y,loc_x,'ro', markersize=7, markeredgewidth=1, markerfacecolor='None')
    for i in range(0, len(loc_x)):
        coord = 'x:' + str(loc_x[i]-10) + '\ny:' + str(loc_y[i]+10) + '\ni:' + str(sfr_mask[loc_x[i],loc_y[i]])
        plt.text( loc_y[i], loc_x[i], coord, color='blue')
        
        
def SFR_apply_cv2(img, loc_x, loc_y, Title='Image', sfr_mask='default',name='SFR_mask',color=(0,0,255)):
    if isinstance(sfr_mask,str):
        sfr_mask = img
        
    if img.ndim == 1:    
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)    
        
    for i in range(0, len(loc_x)):
        cv2.circle( img, (loc_y[i], loc_x[i]), 3, color, -1)
        #coord = 'x:' + str(loc_x[i]-10) + ' y:' + str(loc_y[i]+10) + ' i:' + str(sfr_mask[loc_x[i],loc_y[i]])
        cv2.putText( img, 'x:' + str(loc_x[i]), (loc_y[i]+10,loc_x[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText( img, 'y:' + str(loc_y[i]), (loc_y[i]+10,loc_x[i]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        #cv2.putText( img, 'i:' + str(sfr_mask[loc_x[i],loc_y[i]]), (loc_y[i],loc_x[i]+50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    cv2.imshow(name,img)    
        
def SFR_routine(img, threshold=200, plots =True):
    img_gauss = gauss_blur(img)
    img_SFR = SFR_mask(img_gauss)
    loc_x, loc_y = SFR_locate( img_SFR, threshold )
    SFR_apply(img, loc_x, loc_y, 'Source', img_SFR)
    SFR_apply(img_SFR, loc_x, loc_y, 'img_SFR')
    
def SFR_routine_cv2( img, threshold=200, plots=True, reverse = False):
    img_gauss = gauss_blur(img)
    img_SFR = SFR_maskfft(img_gauss, reverse = reverse)
    loc_x, loc_y = SFR_locate( img_SFR, threshold )
    SFR_apply_cv2(img, loc_x, loc_y, 'Source', img_SFR)
    SFR_apply_cv2(img_SFR, loc_x, loc_y, 'img_SFR')
    
    
def determine_scale(dist, c1, c2):
    #Using two points and a known real distance, find scale ratio
    #    in units of (real units)/pixel
    
    d = math.sqrt( (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
    return (dist/d)



def loc_inrange( loc_x,loc_y, c1,c2 ):
    # Give only points in a given rectangle betwen c1 and c2
    # Make sure c1 is top left and c2 is bottom right
    rloc_x = []
    rloc_y = []
    for i in range(0,len(loc_x)):
        if (loc_x[i] > c1[0]) & (loc_x[i] < c2[0]):
            if (loc_y[i] > c1[1]) & (loc_y[i] <c2[1]):
                rloc_x.append(loc_x[i])
                rloc_y.append(loc_y[i])
    rlocs = [rloc_x, rloc_y]
    return(rlocs)