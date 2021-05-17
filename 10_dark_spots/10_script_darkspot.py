#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yuri tolkach
"""


# =============================================================================
# 10. Dark spots
# =============================================================================


#Parameters
#Directory for files (dataset)
source_dir = '/source/dir/'
#Output directory for results
output_dir = '/output/dir/'
path_result = output_dir + "10_dark_spots.txt"
#Directory with spots
spot_dir = '/spot/dir/spots_small/'
#Model directory (pre-tranied prostate cancer detection model)
model_dir = '/model/dir/'
model_name = 'model.h5'

######Import necessary libraries
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from random import randint


#Load model
path_model = os.path.join(model_dir, model_name)
model = load_model(path_model)
model.summary()

#Function for classification prediction using a model
def predict (patch):
    wp_temp = np.float32(patch)
    wp_temp = np.expand_dims(wp_temp, axis = 0)
    wp_temp /= 255.    
    preds = model.predict(wp_temp)
    return preds

#Function to write result into output txt file
def write_result (output, path_result):
    results = open (path_result, "a+")
    results.write(output)
    results.close()       

#Function to create overlay
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


#Load spots. Coordinates are for a random range of coordinates where spot can be applied to fully affect the  patch
sp_1 = cv2.imread(spot_dir+'SP1.png', cv2.IMREAD_UNCHANGED) #Coordinates range [30:180, 30:180]
sp_2 = cv2.imread(spot_dir+'SP2.png', cv2.IMREAD_UNCHANGED) #Coordinates range [0:150, 0:150]
sp_3 = cv2.imread(spot_dir+'SP3.png', cv2.IMREAD_UNCHANGED) #Coordinates range [-20:130, -20:130]
sp_4 = cv2.imread(spot_dir+'SP4.png', cv2.IMREAD_UNCHANGED) #Coordinates range [-100:0, -100:0]
sp_lst = [sp_1, sp_2, sp_3, sp_4]
x_coord = [30, 0, -20, -100]
y_coord = [180, 150, 130, 0]


#Read subdirectory names
dir_names = sorted(os.listdir(source_dir)) # gland, nongland, tumor

for dir_name in dir_names:
    work_dir = source_dir + dir_name + "/"    
    filenames = os.listdir(work_dir)
        
    #Loop for opening of single files, generating artifact, and
    #making classification predictions.
    for filename in filenames:
        
        preds_all = ''
        preds_all = filename + "\t"
        
        for i in range (0,4,1):
            image = cv2.imread(work_dir+filename)
            he = randint (x_coord[i], y_coord[i])
            he = randint (x_coord[i], y_coord[i])
            wi = randint (x_coord[i], y_coord[i])
            wi = randint (x_coord[i], y_coord[i])
            image_out = transparentOverlay(image,sp_lst[i],(he,wi),1)
            #cv2.imwrite(output_dir + str(i) + "_" + str(he) +"_" + str(wi) + "_" + filename, image_out, [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_ou = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
            preds = predict(image_ou)
            preds_all = preds_all + str(round(preds[0,0],3)) + "\t" + str(round(preds[0,1],3)) + "\t" + str(round(preds[0,2],3)) + "\t"
            he = randint (x_coord[i], y_coord[i])
            he = randint (x_coord[i], y_coord[i])
            wi = randint (x_coord[i], y_coord[i])
            wi = randint (x_coord[i], y_coord[i])
            image_out = transparentOverlay(image,sp_lst[i],(he,wi),1)
            #cv2.imwrite(output_dir + str(i) + "_" + str(he) +"_" + str(wi) + "_" + filename, image_out, [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_ou = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
            preds = predict(image_ou)
            preds_all = preds_all + str(round(preds[0,0],3)) + "\t" + str(round(preds[0,1],3)) + "\t" + str(round(preds[0,2],3)) + "\t"
            he = randint (x_coord[i], y_coord[i])
            he = randint (x_coord[i], y_coord[i])
            wi = randint (x_coord[i], y_coord[i])
            wi = randint (x_coord[i], y_coord[i])
            image_out = transparentOverlay(image,sp_lst[i],(he,wi),1)
            #cv2.imwrite(output_dir + str(i) + "_" + str(he) +"_" + str(wi) + "_" + filename, image_out, [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_ou = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
            preds = predict(image_ou)
            preds_all = preds_all + str(round(preds[0,0],3)) + "\t" + str(round(preds[0,1],3)) + "\t" + str(round(preds[0,2],3)) + "\t"

        preds_all = preds_all + "\n"
        write_result(preds_all, path_result)