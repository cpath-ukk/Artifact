#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yuri tolkach
"""


# =============================================================================
# 12. Squamous epithelia
# =============================================================================


#Parameters
#Directory for files (dataset)
source_dir = '/source/dir/'
#Output directory for results
output_dir = '/output/dir/'
path_result = output_dir + "12_squamous_epithelia.txt"
#Model directory (pre-tranied prostate cancer detection model)
model_dir = '/model/dir/'
model_name = 'model.h5'
#Directory with squamous epithelia
epit_dir = '/media/dr_pusher/YT_SSD3/99_scripts_artefacts/99_work/12_squamous/epit_small/'


######Load necessary libraries
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from random import randint


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


#Load spots
ep_1 = cv2.imread(epit_dir+'sq1.png', cv2.IMREAD_UNCHANGED)
ep_2 = cv2.imread(epit_dir+'sq2.png', cv2.IMREAD_UNCHANGED)
ep_3 = cv2.imread(epit_dir+'sq3.png', cv2.IMREAD_UNCHANGED)
ep_4 = cv2.imread(epit_dir+'sq4.png', cv2.IMREAD_UNCHANGED)
ep_5 = cv2.imread(epit_dir+'sq5.png', cv2.IMREAD_UNCHANGED)
ep_6 = cv2.imread(epit_dir+'sq6.png', cv2.IMREAD_UNCHANGED)
ep_7 = cv2.imread(epit_dir+'sq7.png', cv2.IMREAD_UNCHANGED)
ep_8 = cv2.imread(epit_dir+'sq8.png', cv2.IMREAD_UNCHANGED)
ep_9 = cv2.imread(epit_dir+'sq9.png', cv2.IMREAD_UNCHANGED)
ep_10 = cv2.imread(epit_dir+'sq10.png', cv2.IMREAD_UNCHANGED)
ep_11 = cv2.imread(epit_dir+'sq11.png', cv2.IMREAD_UNCHANGED)
ep_12 = cv2.imread(epit_dir+'sq12.png', cv2.IMREAD_UNCHANGED)
ep_13 = cv2.imread(epit_dir+'sq13.png', cv2.IMREAD_UNCHANGED)
ep_14 = cv2.imread(epit_dir+'sq14.png', cv2.IMREAD_UNCHANGED)
ep_15 = cv2.imread(epit_dir+'sq15.png', cv2.IMREAD_UNCHANGED)
ep_16 = cv2.imread(epit_dir+'sq16.png', cv2.IMREAD_UNCHANGED)
ep_17 = cv2.imread(epit_dir+'sq17.png', cv2.IMREAD_UNCHANGED)
ep_18 = cv2.imread(epit_dir+'sq18.png', cv2.IMREAD_UNCHANGED)
ep_19 = cv2.imread(epit_dir+'sq19.png', cv2.IMREAD_UNCHANGED)
ep_20 = cv2.imread(epit_dir+'sq20.png', cv2.IMREAD_UNCHANGED)

eps = [ep_1, ep_2, ep_3, ep_4, ep_5, ep_6, ep_7, ep_8, ep_9, ep_10, ep_11, ep_12, ep_13, ep_14, ep_15, ep_16, ep_17, ep_18, ep_19, ep_20]

#Read subdirectory names
dir_names = sorted(os.listdir(source_dir)) # gland, nongland, tumor

for dir_name in dir_names:
    work_dir = source_dir + dir_name + "/"    
    filenames = os.listdir(work_dir)
        
    #Generation and analysis loop
    for filename in filenames:

        preds_all = ''
        preds_all = filename + "\t"
        
        image = cv2.imread(work_dir+filename)
        #randomly select type of squamous epithelial complex
        i = randint(0,19)
        #randomly select coordinates at which it should appear (2x)
        he = randint (0,150)
        he = randint (0,150)
        wi = randint (0,150)
        wi = randint (0,150)
        image_out = transparentOverlay(image,eps[i],(he,wi),1)
        #cv2.imwrite(output_dir + str(i) + "_" + str(he) +"_" + str(wi) + "_" + filename, image_out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_ou = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
        preds = predict(image_ou)
        preds_all = preds_all + str(round(preds[0,0],3)) + "\t" + str(round(preds[0,1],3)) + "\t" + str(round(preds[0,2],3)) + "\t"
        i = randint(0,19)
        he = randint (0,150)
        he = randint (0,150)
        wi = randint (0,150)
        wi = randint (0,150)
        image_out = transparentOverlay(image,eps[i],(he,wi),1)
        #cv2.imwrite(output_dir + str(i) + "_" + str(he) +"_" + str(wi) + "_" + filename, image_out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_ou = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
        preds = predict(image_ou)
        preds_all = preds_all + str(round(preds[0,0],3)) + "\t" + str(round(preds[0,1],3)) + "\t" + str(round(preds[0,2],3)) + "\t"
        i = randint(0,19)
        he = randint (0,150)
        he = randint (0,150)
        wi = randint (0,150)
        wi = randint (0,150)
        image_out = transparentOverlay(image,eps[i],(he,wi),1)
        #cv2.imwrite(output_dir + str(i) + "_" + str(he) +"_" + str(wi) + "_" + filename, image_out, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_ou = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
        preds = predict(image_ou)
        preds_all = preds_all + str(round(preds[0,0],3)) + "\t" + str(round(preds[0,1],3)) + "\t" + str(round(preds[0,2],3)) + "\t"

        preds_all = preds_all + "\n"
        write_result(preds_all, path_result)