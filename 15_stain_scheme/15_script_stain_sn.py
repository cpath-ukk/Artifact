#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yuri tolkach
"""


# =============================================================================
# 15. stain scheme
# =============================================================================

#Parameters
#Directory for files (dataset)
source_dir = '/source/dir/'
#Output directory for results
output_dir = '/output/dir/'
path_result = output_dir + "15_stain.txt"
#Model directory (pre-tranied prostate cancer detection model)
model_dir = '/model/dir/'
model_name = 'model.h5'
#Directory with oil drop
#Staintools dir
stain_dir = '/dir/to/stain/schemes/schemes_ready/'

###### Import necessary libraries
import os
import numpy as np
from tensorflow.keras.models import load_model
import staintools
from tensorflow.keras.preprocessing import image

#Load model
path_model = os.path.join(model_dir, model_name)
model = load_model(path_model)
model.summary()

#Model prediction function
def predict (patch):
    wp_temp = np.float32(patch)
    wp_temp = np.expand_dims(wp_temp, axis = 0)
    wp_temp /= 255.    
    preds = model.predict(wp_temp)
    return preds

#Function to write out prediction results
def write_result (output, path_result):
    results = open (path_result, "a+")
    results.write(output)
    results.close()       

#Read subdirectory names
dir_names = sorted(os.listdir(source_dir)) # gland, nongland, tumor

stain_types = sorted(os.listdir(stain_dir))

for stain_type in stain_types:
    st = staintools.read_image(stain_dir + stain_type)
    standardizer = staintools.BrightnessStandardizer()
    stain_norm = staintools.StainNormalizer(method='macenko')
    stain_norm.fit(st)
    
    path_result = path_result_gl + "_" + stain_type + ".txt"
    
    for dir_name in dir_names:
        work_dir = source_dir + dir_name + "/"    
        filenames = os.listdir(work_dir)
            
        #start generating blur and analysis
        for filename in filenames:
            im = image.load_img(work_dir + filename, target_size=(300,300))
            im = np.array(im)
            #stain normalization
            im = standardizer.transform(im)
            
            try:
                im = stain_norm.transform(im)
                i=1
            except:
                print("exception")
                i=0 #to control if stain transfer was possible for all patches
            print("ready")

            #prediction
            preds = predict(im)
            print("ready")
            
            preds_all = ''
            preds_all = filename + "\t" + str(i) + "\t"
            preds_all = preds_all + str(round(preds[0,0],3)) + "\t" + str(round(preds[0,1],3)) + "\t" + str(round(preds[0,2],3)) + "\t"
            preds_all = preds_all + "\n"
            write_result(preds_all, path_result)
            


