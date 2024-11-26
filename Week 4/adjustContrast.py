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
    
    # Convert to uint8, ensuring valid range [0, 255]
    image_uint8 = np.clip(image.astype(np.float32), 0, 255).astype(np.uint8)

    # outImg = image_uint8
    
    # # Find min and max values
    min_val = np.min(image_uint8)
    max_val = np.max(image_uint8)
    
    # # Stretch contrast using the formula
    stretched_image = (image_uint8 - min_val) * (255.0 / (max_val - min_val))
    
    # # Ensure values are clipped and cast to uint8
    # outImg = np.clip(stretched_image, 0, 255).astype(np.uint8)
    outImg = stretched_image



    # END OF YOUR CODE
    #########################################################################
    return outImg