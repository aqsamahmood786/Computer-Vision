# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:40:46 2020

@author: aqsa
"""
import glob 
import os
import cv2
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
from untilities import write_image_to_directory
from untilities import gaussian_filter
from untilities import convertToRGB


class DataAugmentation:

    def augment(self,image_source_directory):
        for root, dirs, files in os.walk(image_source_directory):
            filtered_files = glob.glob(os.path.join(root,'*_VIDEO_FRAME_*.jpg'))
            for i in  range(len(filtered_files)):
                file = filtered_files[i]
                if((i+1)%6==0):
                    file_path = os.path.join(root, file)
                    file_name = os.path.basename(file_path)
                    label = os.path.basename(root)
                    
                    img = cv2.imread(file_path)  
                    #img = noise(gaussian_filter(img))
                    #img = gaussian_filter(img)        
                    samples = expand_dims(img, 0)
                    # create image data augmentation generator
                    datagen = ImageDataGenerator(zoom_range=0.4,shear_range=5,rotation_range=5,brightness_range=[0.9,1.5],horizontal_flip=True)
                    # prepare iterator
                    it = datagen.flow(samples, batch_size=1)
                    # generate samples and plot
                    for i in range(9):
                    	# generate batch of images
                        batch = it.next()
                    	# convert to unsigned integers for viewing
                        image = convertToRGB(batch[0].astype('uint8'))
                        #print(filename+'_augmentation_{}.jpg'.format(i))
                        write_image_to_directory(os.path.splitext(file_name)[0]+'_augmentation_{}.jpg'.format(i),os.path.join(image_source_directory,label),image)
    def remove_augmented_files(self,image_source_directory):
        for root, dirs, files in os.walk(image_source_directory):
            filtered_files = glob.glob(os.path.join(root,'*_VIDEO_FRAME_augmentation_*.jpg'))
            for i in  range(len(filtered_files)):
                file = filtered_files[i]
                os.remove(os.path.join(root, file))