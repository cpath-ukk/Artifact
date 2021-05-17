#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yuri tolkach
"""

# =============================================================================
# 1. Focus artefacts (Gaussian blur)
# =============================================================================

#Parameters
#Number of gaussian levels to test
num_g_lev = 16
#Directory for files (dataset)
source_dir = '/source/dir/'
#Output directory for results
output_dir = '/output/dir/'
path_result = output_dir + "01_focus.txt"
#Model directory (pre-tranied prostate cancer detection model)
model_dir = '/model/dir/'
model_name = 'model.h5'


#Load necessary libraries
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

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

#Read subdirectory names
dir_names = sorted(os.listdir(source_dir)) # gland, nongland, tumor

for dir_name in dir_names:
    work_dir = source_dir + dir_name + "/"    
    filenames = os.listdir(work_dir)
        
    #Loop for opening of single files, generating artifact, and
    #making classification predictions.
    for filename in filenames:
        image = cv2.imread(work_dir+filename)
        image = cv2.resize(image, (300, 300), cv2.INTER_AREA)
        print('loaded', filename)
        preds_all = ''
        preds_all = filename + "\t"
        for i in range(1,2*num_g_lev,2):
            image_blur = cv2.GaussianBlur(image, (i, i), 0)
            image_blur = cv2.cvtColor(image_blur, cv2.COLOR_BGR2RGB)
            preds = predict(image_blur)
            preds_all = preds_all + str(round(preds[0,0],3)) + "\t" + str(round(preds[0,1],3)) + "\t" + str(round(preds[0,2],3)) + "\t"
        preds_all = preds_all + "\n"
        write_result(preds_all, path_result)
            


