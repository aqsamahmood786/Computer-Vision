# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:49:53 2020

@author: aqsa
"""
import numpy as np
import os
import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import torch
from numpy import expand_dims
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

from untilities import convertToRGB
from untilities import write_image_to_directory
from untilities import create_directory



class FaceExtracter:
    
    def __init__(self, image_source_directory,labelled_dir,unlabelled_dir):
        self.image_source_directory = image_source_directory

        # Create directory 'dataset' if it does not exist
        self.labelled_dir = labelled_dir
        
        #unlabelled directory for exracted faces from group images/videos
        self.unlabelled_dir = unlabelled_dir

        create_directory(self.labelled_dir)
        create_directory(self.unlabelled_dir)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

    def extract(self,thresholds=[0.8, 0.9, 0.9],margin=20,image_size=224):                   
        for root, dirs, files in os.walk(self.image_source_directory):
            for file in files:
                
                file_path = os.path.join(root, file)
                label = os.path.basename(root)
                file_name = os.path.basename(file_path)
                frame_sample_threshold = 30
                
                faces_dir = self.labelled_dir
                if (label == 'group'):
                    ##save extracted faces fom group images to un-labelled directory
                    faces_dir = self.unlabelled_dir
                    frame_sample_threshold = 20
                # Create face detector
                faces = None
                mtcnn = MTCNN(image_size=image_size, keep_all=True,margin=margin, post_process=False, device=self.device,thresholds=thresholds)
                file_name_no_ext,ext = os.path.splitext(file_name)
                face_fle_path = None
                
                if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
                    pixels = convertToRGB(cv2.imread(file_path))
                    face_fle_path = os.path.join(faces_dir,label,file_name_no_ext )
                    print(face_fle_path+ext)
                    faces = mtcnn(pixels,save_path=face_fle_path+ext)
                elif file.endswith('.mov') or file.endswith('.mp4'):
                    
                    vid_capture = cv2.VideoCapture(file_path)
                    success,image = vid_capture.read()

                    v_len = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    frames = [] # to extract some frames from video
                    
                    for i in range(v_len):
                        success, image = vid_capture.read()
                                            
                        if len(frames) > frame_sample_threshold or not success: # if video is too long stop after sampling 20 frames
                            break
                        
                        if ((i+1)%3 == 0) & success:
                            image = convertToRGB(image)
                            frames.append(Image.fromarray(image))
        
                     # Detect faces in batch
                    
                    extracted_image_video_file_name = file_name_no_ext+'_'+'video_frame'.upper()
                    face_fle_path = os.path.join(faces_dir,label,extracted_image_video_file_name) 
                    save_paths = [face_fle_path+f'_{i}.jpg' for i in range(len(frames))]
                    faces = mtcnn(frames, save_path=save_paths)   
                #release cuda memory
                del mtcnn
                torch.cuda.empty_cache()
                
                #if (label != 'group'):
                    #self.image_augmentation(faces,face_fle_path)