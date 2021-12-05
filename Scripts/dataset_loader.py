# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:01:56 2020

@author: aqsa
"""
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from untilities import convertToRGB
import cv2


class DatasetLoader:
    def __init__(self, directory):
        self.directory = directory
       
    def load(self):
        ##loading extracted faces dataset
        images = []
        labels = []
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_name = os.path.basename(file_path)
                img = cv2.imread(file_path);
                if (img is None):
                    continue
                images.append(convertToRGB(img))
                labels.append(os.path.basename(root))
                
        images, labels = shuffle(images, labels)


        return images,labels

