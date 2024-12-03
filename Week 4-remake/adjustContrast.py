# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:19:02 2024

@author: MU180926
"""

import cv2
import numpy as np


input_dir = 'dataset/test'
output_dir = 'dataset/output'

# you are allowed to import other Python packages above
##########################
def adjustContrast(img):
    # Inputs
    # img: Input image, a 3D numpy array of row*col*3 in BGR format
    #
    # Output
    # outImg: enhanced image
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    image_uint8 = np.clip(image.astype(np.float32), 0, 255).astype(np.uint8)

    min_val = np.min(image_uint8)
    max_val = np.max(image_uint8)
    

    stretched_image = (image_uint8 - min_val) * (255.0 / (max_val - min_val))
    
    outImg = stretched_image

    
    
    
     
    # END OF YOUR CODE
    #########################################################################
    return outImg