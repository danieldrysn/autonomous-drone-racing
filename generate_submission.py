# Load libraries
import json
import ntpath
import os
import sys
from pprint import pprint
import glob
import cv2
import numpy as np
from random import shuffle
from generate_results import *
import time
import argparse


# Parse arguments
parser = argparse.ArgumentParser(description='Generates .json file for gate detection predictions')
parser.add_argument('weights_path', help='file path to weights file')
parser.add_argument('image_directory', help='directory to images')
parser.add_argument('json_path', help='file path to output .json file')

args = parser.parse_args()


# Retrieve images
img_files = glob.glob(os.path.join(args.image_directory, '*.JPG'))


# Instantiate a new detector
finalDetector = GenerateFinalDetections(args.weights_path)


# load image, convert to RGB, run model and plot detections. 
time_all = []
pred_dict = {}
for img_file in img_files:
    img =cv2.imread(img_file)
    img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tic = time.monotonic()
    bb_all = finalDetector.predict(img)
    toc = time.monotonic()
    pred_dict[ntpath.basename(img_file)] = bb_all
    time_all.append(toc-tic)

mean_time = np.mean(time_all)
ci_time = 1.96*np.std(time_all)
freq = np.round(1/mean_time,2)
    
print('95% confidence interval for inference time is {0:.2f} +/- {1:.4f}.'.format(mean_time,ci_time))
print('Operating frequency from loading image to getting results is {0:.2f}.'.format(freq))


# Write to json file
with open(args.json_path, 'w') as f:
    json.dump(pred_dict, f)
