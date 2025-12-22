#feature-level preprocessing

# resizing images for feature extraction, coverting to grayscale for HOG, normalizing histograms, ensuring fixed feature length. 

import numpy as np
import cv2 as cv 
from skimage.feature import hog

def make_consistent(img, size= (224, 224)):
    return cv.resize(img, size)

def extract_color_hist(img, bins = 16):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) #bgr default to hsv

    features = []

    for hsv_channel in range(3):
        hist = cv.calcHist([img_hsv], [hsv_channel], None, [bins], [0, 256])

        features.append(hist.flatten()) #2D to 1D array

    features = np.concatenate(features) #combine all features
    features = features/ (features.sum() + 1e-8) #normalization 

    return features

#HOG, brightness changing across images (intensity)

def extract_hog_features(img_gray):

    hog_features = hog( 
        img_gray,
        orientations = 9,
        pixels_per_cell = (8,8),
        cells_per_block= (2,2),
        block_norm = 'L2-Hys', #scaling rule, reduces the effect of lighting difference
        transform_sqrt = True,
        feature_vector = True,
    )
    return hog_features

#combine color features and HOG features to one feature vector

def extract_features(img):
    img_resize = cv.resize(img, (224, 224))
    img_gray = cv.cvtColor(img_resize, cv.COLOR_BGR2GRAY) #convert color

    color_features = extract_color_hist(img_resize)
    hog_features = extract_hog_features(img_gray)
    
    features = np.concatenate([color_features, hog_features])

    return features