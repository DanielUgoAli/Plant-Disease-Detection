#Taking features and learn how to map them to label

import numpy as np
import os
import cv2 as cv

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing  import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

from .features import extract_features

def build_dataset(image_root, max_per_class= 200): #set a threshold for testing
    X = []
    y = []
   

    for class_name in os.listdir(image_root):
        class_path = os.path.join(image_root, class_name) #filepath 
        if not os.path.isdir(class_path):
            continue

        count = 0
    
        for image_name in os.listdir(class_path):
            if count>= max_per_class:
                break

            img_path = os.path.join(class_path, image_name) #python finds the image
            if not os.path.isfile(img_path): #skip what isn't a file
                continue

            img = cv.imread(img_path)
            if img is None:
                continue
            
            features = extract_features(img) 
            X.append(features)
            y.append(class_name)
            count += 1
    
    return X, y #return value