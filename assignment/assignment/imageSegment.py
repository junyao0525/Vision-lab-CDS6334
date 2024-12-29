# -*- coding: utf-8 -*-
"""
imageSegment.py

YOUR WORKING FUNCTION

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt



input_dir = 'dataset/test'
output_dir = 'dataset/output'
gt_dir = 'dataset/groundtruth'

# you are allowed to import other Python packages above
##########################



def segmentImage(img):
    # Inputs
    # img: Input image, a 3D numpy array of row*col*3 in BGR format
    #
    # Output
    # outImg: segmentation image
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE

    def remove_circle (img) :

        greyScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(greyScale, 1, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a mask for the circular region
        mask = np.zeros_like(greyScale)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        # Invert the mask to remove the circular boundary
        inverted_mask = cv2.bitwise_not(mask)

        kernel = np.ones((3, 3), np.uint8)
        inverted_mask = cv2.dilate(inverted_mask, kernel,iterations=1)

        return inverted_mask

    green_channel = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[1]

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(green_channel)

    # Apply median blur to reduce noise
    filtered_img = cv2.GaussianBlur(enhanced_img,(5,5),1) 

    outImg = cv2.adaptiveThreshold(
        filtered_img,
        maxValue=1,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=5,
    )        

    # Post-processing to remove small regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(outImg, connectivity=8)
    min_size = 45
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            outImg[labels == i] = 0

    # Remove the circular region
    remove_circle_img = remove_circle(img)

    outImg = cv2.subtract(outImg, remove_circle_img)

    # END OF YOUR CODE
    #########################################################################
    return outImg

