# -*- coding: utf-8 -*-
"""
imageSegment.py

YOUR WORKING FUNCTION

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


input_dir = 'dataset/test'
output_dir = 'dataset/output'
gt_dir = 'dataset/groundtruth'

# you are allowed to import other Python packages above
##########################

def apply_CLAHE(img, clip_limit, tile_grid_size=(15, 15)):
    # Apply CLAHE to enhance local contrast
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def gaussian_blur(img, kernel_size, sigma):
    # Apply Gaussian smoothing
    return cv2.GaussianBlur(img, kernel_size, sigma)

def compute_gradients(img):
    # Calculate Sobel gradients
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    # Calculate gradient magnitude and direction
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = np.arctan2(grad_y, grad_x) * 180 / np.pi

    # Normalize gradient magnitude
    # magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return magnitude, direction


def non_maximum_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z



def double_threshold(img, low_threshold, high_threshold):
    # Apply double threshold
    strong = np.zeros_like(img, dtype=np.uint8)
    weak = np.zeros_like(img, dtype=np.uint8)
    strong[img >= high_threshold] = 255
    weak[(img >= low_threshold) & (img < high_threshold)] = 128
    return strong, weak

def hysteresis(strong, weak):
    # Apply hysteresis to track edges
    final_edges = np.copy(strong)
    h, w = strong.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if weak[i, j] == 128:
                if np.any(strong[i - 1:i + 2, j - 1:j + 2] == 255):
                    final_edges[i, j] = 255
                else:
                    final_edges[i, j] = 0
    return final_edges

def remove_fundus_outline(canny_img, green_channel):
    # Create binary mask of fundus outline
    _, binary_mask = cv2.threshold(green_channel, 15, 255, cv2.THRESH_BINARY)
    eroded_mask = cv2.erode(binary_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.bitwise_and(canny_img, eroded_mask)

def sharpening_img(img ,size=5):
    
    blurrer_img = cv2.GaussianBlur(img,(size,size),0)

    sm_img = cv2.blur(blurrer_img,(size,size)).astype(np.float32)


    detail_img = blurrer_img - sm_img
    outImg = np.clip(blurrer_img + detail_img, 0, 255).astype(np.uint8)


    return outImg


def segmentImage(img):
    # Inputs
    # img: Input image, a 3D numpy array of row*col*3 in BGR format
    #
    # Output
    # outImg: segmentation image
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE

    green_channel = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[1]
    clahe_img = apply_CLAHE(green_channel, 2, (13, 13))
    blurred = gaussian_blur(clahe_img, (5, 5), 1.4)
    blurred = sharpening_img(blurred)
    magnitude, direction = compute_gradients(blurred)
    suppressed = non_maximum_suppression(magnitude, direction)

    high_threshold = np.percentile(suppressed, 90)
    low_threshold = 0.3 * high_threshold
    strong, weak = double_threshold(suppressed, low_threshold, high_threshold)
    edges = hysteresis(strong, weak)
    kernel = np.ones((5, 5), np.uint8)
    outImg = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Convert to binary mask: Normalize to [0, 1]
    binary_mask = (outImg > 0).astype('float32')  # Ensure binary (0 or 1) and float32 format

    return binary_mask
   
    # END OF YOUR CODE
    #########################################################################
    return outImg
