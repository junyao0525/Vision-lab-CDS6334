# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:20:05 2024

@author: MU180926
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import importlib
import sys, getopt
from prettytable import PrettyTable
import adjustContrast as adj
from os import listdir
from os.path import isfile, join, splitext
from skimage.metrics import structural_similarity as SSIM

# Default parameters (the only code you can change)
verbose = False #False, 1, or 2
input_dir = 'dataset/test'
output_dir = 'dataset/output'
groundtruth_dir = 'dataset/groundtruth'
numImages = 4
eps = 0.00000001

onlyfiles = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
files = onlyfiles[0:numImages]

## Read command linehargs
myopts, args = getopt.getopt(sys.argv[1:],"i:vph")

# Reload module
importlib.reload(adj)

################################################
# o == option    a == argument passed to the o #
################################################

# parsing command line args
for o, a in myopts:
    #print(o)
    #print(a)
    if o == '-v':
        verbose = 1
    elif o == '-h':
        print("\nUsage: %s -v to show evaluation for every image" % sys.argv[0])
        sys.exit()
    else:
        print(' ')

mae = np.zeros(numImages)
psnr = np.zeros(numImages)
ssim = np.zeros(numImages)

# Evaluation function
def PSNR(original, compressed):
    MSE = np.mean((original - compressed) ** 2)
    if(MSE == 0):  # MSE is zero means no noise is present in the signal and PSNR should be infinity.
        return 100 # Place a dummy value to avoid division by zero
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(MSE))
    return psnr


# Evaluate each image and compare with ground-truth
for i,name in enumerate(files):
    inputImg = cv2.imread(input_dir + '/' + name)
    outputImg = adj.adjustContrast(inputImg).astype('float32')
    imgName = splitext(name)
    # outputImg = cv2.cvtColor(outputImg,cv2.COLOR_BGR2RGB)
    plt.imsave(output_dir + '/' + imgName[0] + '.jpg',outputImg.astype('uint8'))
    gt = cv2.imread(groundtruth_dir + '/' + imgName[0] + '.jpg')
    gt = np.round(gt.astype('float32')/255)
    
    mae[i] = np.mean(np.abs(gt - outputImg))              
    psnr[i] = PSNR(gt,outputImg)
    ssim[i] = SSIM(cv2.cvtColor(gt,cv2.COLOR_RGB2GRAY),
                   cv2.cvtColor(outputImg,cv2.COLOR_RGB2GRAY),data_range=outputImg.max() - outputImg.min())

# Print performance scores        
if verbose==1:
    print('####  IMAGE RESULTS  ####')
    t = PrettyTable(['Image', 'MAE','PSNR','SSIM'])
    avg_mae = np.mean(mae)
    avg_psnr = np.mean(psnr)
    avg_ssim = np.mean(ssim)
    
    
    for i in range(numImages):
        t.add_row([i+1, str(round(mae[i],4)),str(round(psnr[i],4)),\
                   str(round(ssim[i],4))]) 
                   
    t.add_row([' ',' ',' ',' '])
    t.add_row(['All', str(round(avg_mae,4)),str(round(avg_psnr,4)),\
               str(round(avg_ssim,4))])
    print(t)
    
else:
    
    
    print('MAE: %.4f' % (np.sum(mae)/numImages))
    print('PSNR: %.4f' % (np.sum(psnr)/numImages))
    print('SSIM: %.4f' % (np.sum(ssim)/numImages))
        
        
# END OF EVALUATION CODE####################################################
