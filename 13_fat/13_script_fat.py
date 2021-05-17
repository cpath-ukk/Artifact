#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yuri tolkach
"""



# =============================================================================
# 13. Greasy fingerprints
# =============================================================================


#Parameters
#Directory for files (dataset)
source_dir = '/source/dir/'
#Output directory for results
output_dir = '/output/dir/'
path_result = output_dir + "13_fingerprint.txt"
#Model directory (pre-tranied prostate cancer detection model)
model_dir = '/model/dir/'
model_name = 'model.h5'
#Directory with oil drop
fat_dir = '/dir/to/fat/'


######Necessary libraries
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

#Load model
path_model = os.path.join(model_dir, model_name)
model = load_model(path_model)
model.summary()

#Prediction function
def predict (patch):
    wp_temp = np.float32(patch)
    wp_temp = np.expand_dims(wp_temp, axis = 0)
    wp_temp /= 255.    
    preds = model.predict(wp_temp)
    return preds

#Write results to file function
def write_result (output, path_result):
    results = open (path_result, "a+")
    results.write(output)
    results.close()       

#Making overlay function
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

#Load oil drop image
fat = cv2.imread(fat_dir+'fat.png', cv2.IMREAD_UNCHANGED)

#Read subdirectory names
dir_names = sorted(os.listdir(source_dir)) # gland, nongland, tumor

for dir_name in dir_names:
    work_dir = source_dir + dir_name + "/"    
    filenames = os.listdir(work_dir)
        
    #start generating and making predictions
    for filename in filenames:
        image = cv2.imread(work_dir+filename)
        #image = cv2.resize(image, (300, 300), cv2.INTER_AREA)
        print('loaded', filename)
        preds_all = ''
        preds_all = filename + "\t"
        image_out = transparentOverlay(image,fat,(0,0),1)
        #cv2.imwrite(output_dir + str(i) + "_" + str(he) +"_" + str(wi) + "_" + filename, image_out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_ou = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
        preds = predict(image_ou)
        preds_all = preds_all + str(round(preds[0,0],3)) + "\t" + str(round(preds[0,1],3)) + "\t" + str(round(preds[0,2],3)) + "\t"
        
        preds_all = preds_all + "\n"
        write_result(preds_all, path_result)