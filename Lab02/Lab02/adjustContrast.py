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
    # outImg: segmentation image
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE
    
    r,g,b =cv2.split(img)
    
    r = cv2.equalizeHist(r)
    g = cv2.equalizeHist(g)
    b = cv2.equalizeHist(b)

    outImg = cv2.merge((r,g,b))
    
    
     
    # END OF YOUR CODE
    #########################################################################
    return outImg