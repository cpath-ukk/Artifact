#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yuri tolkach
"""


# =============================================================================
# 11. Synthetic threads with local focus deterioration
# =============================================================================


#Parameters
#Directory for files (dataset)
source_dir = '/source/dir/'
#Output directory for results
output_dir = '/output/dir/'
path_result = output_dir + "11_threads.txt"
#Model directory (pre-tranied prostate cancer detection model)
model_dir = '/model/dir/'
model_name = 'model.h5'
#Directory with threads
thread_dir = '/thread/dir/thread_small/'


######Load necessary libraries
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from random import randint
from PIL import Image


#Load model
path_model = os.path.join(model_dir, model_name)
model = load_model(path_model)
model.summary()

#Function for model predictions
def predict (patch):
    wp_temp = np.float32(patch)
    wp_temp = np.expand_dims(wp_temp, axis = 0)
    wp_temp /= 255.    
    preds = model.predict(wp_temp)
    return preds

#Function to write the predictions to the output file
def write_result (output, path_result):
    results = open (path_result, "a+")
    results.write(output)
    results.close()       

#Function to make overlay
def transparentOverlay(src, overlay, pos=(0,0), scale = 1):
    overlay = cv2.resize(overlay,(0,0),fx=scale,fy=scale)
    h,w,_ = overlay.shape  # Size of foreground
    rows,cols,_ = src.shape  # Size of background Image
    y,x = pos[0],pos[1]    # Position of foreground/overlay image
    
    #loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x+i >= rows or y+j >= cols:
                continue
            alpha = float(overlay[i][j][3]/255.0) # read the alpha channel 
            src[x+i][y+j] = alpha*overlay[i][j][:3]+(1-alpha)*src[x+i][y+j]
    return src


#Load threads
thread1 = cv2.imread(thread_dir+'thread_1.png', cv2.IMREAD_UNCHANGED)
thread2 = cv2.imread(thread_dir+'thread_2.png', cv2.IMREAD_UNCHANGED)
thread3 = cv2.imread(thread_dir+'thread_3.png', cv2.IMREAD_UNCHANGED)
thread4 = cv2.imread(thread_dir+'thread_4.png', cv2.IMREAD_UNCHANGED)
thread5 = cv2.imread(thread_dir+'thread_5.png', cv2.IMREAD_UNCHANGED)
thread6 = cv2.imread(thread_dir+'thread_6.png', cv2.IMREAD_UNCHANGED)
thread7 = cv2.imread(thread_dir+'thread_7.png', cv2.IMREAD_UNCHANGED)
thread8 = cv2.imread(thread_dir+'thread_8.png', cv2.IMREAD_UNCHANGED)
thread9 = cv2.imread(thread_dir+'thread_9.png', cv2.IMREAD_UNCHANGED)
thread10 = cv2.imread(thread_dir+'thread_10.png', cv2.IMREAD_UNCHANGED)

#Load masks
mask1 = np.array(Image.open(thread_dir+'thread_1_mask.jpg'))
mask2 = np.array(Image.open(thread_dir+'thread_2_mask.jpg'))
mask3 = np.array(Image.open(thread_dir+'thread_3_mask.jpg'))
mask4 = np.array(Image.open(thread_dir+'thread_4_mask.jpg'))
mask5 = np.array(Image.open(thread_dir+'thread_5_mask.jpg'))
mask6 = np.array(Image.open(thread_dir+'thread_6_mask.jpg'))
mask7 = np.array(Image.open(thread_dir+'thread_7_mask.jpg'))
mask8 = np.array(Image.open(thread_dir+'thread_8_mask.jpg'))
mask9 = np.array(Image.open(thread_dir+'thread_9_mask.jpg'))
mask10 = np.array(Image.open(thread_dir+'thread_10_mask.jpg'))

masks = [mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9, mask10]
threads = [thread1, thread2, thread3, thread4, thread5, thread6, thread7, thread8, thread9, thread10]


#Subdirectory names
dir_names = sorted(os.listdir(source_dir)) # gland, nongland, tumor

for dir_name in dir_names:
    work_dir = source_dir + dir_name + "/"    
    filenames = os.listdir(work_dir)
        
    #Processing single files
    for filename in filenames:
        image = cv2.imread(work_dir+filename)
        #image = cv2.resize(image, (300, 300), cv2.INTER_AREA)
        print('loaded', filename)
        preds_all = ''
        preds_all = filename + "\t"
        
        i = randint (0, 9)
        
        image_over = transparentOverlay(image,threads[i],(0,0),1)
        
        image_blur = cv2.GaussianBlur(image_over, (21, 21), 0)

        # (0,0,0) is 3 dimensions of numpy and for every dimension 0 is threshold to be masked
        image_out = np.where(masks[i]==(0, 0, 0), image_over, image_blur)
        
        #cv2.imwrite(output_dir + filename, image_out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        
        image_out = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
        preds = predict(image_out)
        preds_all = preds_all + str(round(preds[0,0],3)) + "\t" + str(round(preds[0,1],3)) + "\t" + str(round(preds[0,2],3)) + "\t"
           
        preds_all = preds_all + "\n"
        write_result(preds_all, path_result)