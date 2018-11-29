
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os




image_shape = (32,32)


def image(folder,file):
    # First put the correct number of zeros in the image string
    fileStr = str(file)
    if len(fileStr) == 1:
        fileStr = '0000' + fileStr + '.jpg'
    else:
        fileStr = '000' + fileStr + '.jpg'
        
    location = 'swipe-videos//' + str(folder) + '//' + fileStr
    im = cv2.imread(location)
    return im

def grayScale(im):
    return np.dot(im,np.asarray([0.2,0.4,0.4]))

def imProcess(im):    
    gray = grayScale(im)
    return cv2.resize(gray,image_shape)/255

def readAndProcess(gesture_No,image_No):
    return imProcess(image(gesture_No,image_No))

# This function reads in the middle frame of a gesture

def readMiddleImage(gesture_no):
    frames = len(os.listdir('swipe-videos/'+str(gesture_no)))
    return readAndProcess(gesture_no,int(frames/2))


# Read in 40 images of a gesture in the form of a 3d array

def gest3D(gesture_No):
    imagesArray = np.zeros((40,) + image_shape)
    location = 'swipe-videos//' + str(gesture_No)
    
    # the number of images in the folder
    
    numImages = len(os.listdir(location))
    duplicates = max(0,40-numImages)
    for i in range(duplicates):
        imagesArray[i] = readAndProcess(gesture_No,1)
        
    for i in range(40-duplicates):
        imagesArray[i+duplicates] = readAndProcess(gesture_No,i+1)
        
    return imagesArray



def label2vec(label):
    result = np.zeros(4)
    direction = label[8:]
    if direction == 'Right':
        result[0] = 1
    elif direction == 'Left':
        result[1] = 1
    elif direction == 'Up':
        result[2] = 1
    elif direction == 'Down':
        result[3] =1
        
    return result

def label2num(label):
    direc = label[8:]
    if direc == 'Right':
        return 0
    elif direc == 'Left':
        return 1
    elif direc == 'Up':
        return 2
    elif direc == 'Down':
        return 3



