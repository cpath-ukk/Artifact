#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yuri tolkach
"""


# =============================================================================
# 8. Brightness DOWN
# =============================================================================


#Parameters
#Directory for files (dataset)
source_dir = '/source/dir/'
#Output directory for results
output_dir = '/output/dir/'
path_result = output_dir + "08_bright_down.txt"
#Model directory (pre-tranied prostate cancer detection model)
model_dir = '/model/dir/'
model_name = 'model.h5'

#Load necessary libraries
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageEnhance

#Load model
path_model = os.path.join(model_dir, model_name)
model = load_model(path_model)
model.summary()

def predict (patch):
    wp_temp = np.float32(patch)
    wp_temp = np.expand_dims(wp_temp, axis = 0)
    wp_temp /= 255.    
    preds = model.predict(wp_temp)
    return preds
  
def write_result (output, path_result):
    results = open (path_result, "a+")
    results.write(output)
    results.close()       

def Brighter (image, factor):
    image_enh = ImageEnhance.Brightness(image)
    image_out = image_enh.enhance(factor)
    return image_out


#Subdirectory names
dir_names = sorted(os.listdir(source_dir)) # gland, nongland, tumor

for dir_name in dir_names:
    work_dir = source_dir + dir_name + "/"    
    filenames = os.listdir(work_dir)
        
    #Loop
    for filename in filenames:
        image = Image.open(work_dir+filename)
        image = image.resize((300, 300), Image.ANTIALIAS)
        print('loaded', filename)
        preds_all = ''
        preds_all = filename + "\t"
        for c in range(1,10,1): #4 levels
            c_cor = 1 - (c / 10) 
            image_br = Brighter(image, c_cor)
            #image_br.save(output_dir + filename + str(c) + ".jpg", quality=80)
            preds = predict(image_br)
            preds_all = preds_all + str(round(preds[0,0],3)) + "\t" + str(round(preds[0,1],3)) + "\t" + str(round(preds[0,2],3)) + "\t"
        preds_all = preds_all + "\n"
        write_result(preds_all, path_result)


