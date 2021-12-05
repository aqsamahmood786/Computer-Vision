# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 19:00:17 2020

@author: aqsa
"""

import numpy as np
import os
import pickle
import bz2
import torch


from sklearn.model_selection import train_test_split
import enum
import cv2
from untilities import create_directory
from untilities import convertToRGB
from untilities import write_to_pickle_file
from untilities import write_to_pickle_bz2file
from untilities import cartoonify
from untilities import add_glasses
from matplotlib import pyplot as plt
from dataset_loader import DatasetLoader
from face_extractor import FaceExtracter
from data_augmentation import DataAugmentation
from loaders import load_cnn
from loaders import load_mlp_model2
from loaders import load_logistic_regression
from loaders import load_svm
from torch import Tensor
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset,Dataset
from facenet_pytorch import MTCNN

#import pytesseract

""" ########################### loading dataset and pre=processing ######################################################### """
base_dir = os.path.dirname(os.getcwd())
resources_absolute_path = os.path.join(base_dir, 'resources')
image_dir = os.path.join(resources_absolute_path, 'samples')

model_outputs_dir = os.path.join(resources_absolute_path,'model_outputs')


if(not os.path.exists(model_outputs_dir)):
    create_directory(model_outputs_dir)
    
#directory which contains creative assets
createive_mode_asssts_dir = os.path.join(resources_absolute_path,'creative_mode_assets')  

# Create directory 'dataset' if it does not exist
labelled_dir = os.path.join(resources_absolute_path,'dataset', 'labelled')
        
#unlabelled directory for exracted faces from group images/videos
unlabelled_dir = os.path.join(resources_absolute_path, 'dataset','unlabelled')

     
  

##Pre-processing, detecting and extracting faces
if(not os.path.exists(os.path.join(resources_absolute_path, 'dataset'))):
    extracter = FaceExtracter(image_dir,labelled_dir,unlabelled_dir)
    extracter.extract()
    augmenter = DataAugmentation()
    augmenter.augment(labelled_dir)
    #augmenter.remove_augmented_files(labelled_dir)

#loading extracted faces and labels
dataset = DatasetLoader(labelled_dir)
images,labels = dataset.load()

images_array = np.asarray(images)
labels_array = np.asarray(labels)

del images,labels


X_train, X_test, y_train, y_test = train_test_split(images_array, labels_array, test_size=0.2, random_state=1)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    
class FeatureType(enum.Enum): 
    SIFT = 1
    SURF = 2
    ORB = 3
class ClassifierType(enum.Enum): 
    SVM = 1
    MLP = 2
    CNN_ResNet = 3
    CNN_vgg16 = 4
    LOG_REGRESSION = 5
    
class CreativeModeType(enum.Enum): 
    cartoonify = 1
    sunglasses_cigar = 2



class FeatureClassifierPair:
     def __init__(self,classifier_type,feature_type):
         self.classifier_type = classifier_type
         self.feature_type = feature_type

     def __eq__(self, other):
         return self.classifier_type == other.classifier_type and self.feature_type == other.feature_type
     
     def __repr__(self):
         return 'FeatureClassifierPair({},{})'.format(self.classifier_type.name,self.feature_type.name if self.feature_type else 'None')        
     def __str__(self):
         return '{}_{}'.format(self.classifier_type.name,self.feature_type.name  if self.feature_type else 'None')
"""##########################################################SVM Loading ###############################################################"""

feature_classifier_map = {}

feature_type = FeatureType.SIFT.name
svm_file_path = os.path.join(model_outputs_dir,'svm_{}_training.pkl'.format(feature_type.lower()))
svm_sift = load_svm(feature_type,svm_file_path,model_outputs_dir,X_train,y_train,X_test,y_test,hyperparameter_tuning=False)
feature_classifier_map[str(FeatureClassifierPair(ClassifierType.SVM,FeatureType.SIFT))] = svm_sift


feature_type = FeatureType.SURF.name
svm_file_path = os.path.join(model_outputs_dir,'svm_{}_training.pkl'.format(feature_type.lower()))
svm_surf= load_svm(feature_type,svm_file_path,model_outputs_dir,X_train,y_train,X_test,y_test,hyperparameter_tuning=False)

feature_classifier_map[str(FeatureClassifierPair(ClassifierType.SVM,FeatureType.SURF))] = svm_surf


feature_type = FeatureType.ORB.name
svm_file_path = os.path.join(model_outputs_dir,'svm_{}_training.pkl'.format(feature_type.lower()))
svm_orb= load_svm(feature_type,svm_file_path,model_outputs_dir,X_train,y_train,X_test,y_test,hyperparameter_tuning=False)

feature_classifier_map[str(FeatureClassifierPair(ClassifierType.SVM,FeatureType.ORB))] = svm_orb

"""##########################################################LOGISTIC REGRESSION Loading ###############################################################"""



feature_type = FeatureType.SIFT.name
log_reg_file_path = os.path.join(model_outputs_dir,'logistic_regression_{}_training.pkl'.format(feature_type.lower()))
log_reg_sift = load_logistic_regression(feature_type,log_reg_file_path,model_outputs_dir,X_train,y_train,X_test,y_test)
feature_classifier_map[str(FeatureClassifierPair(ClassifierType.LOG_REGRESSION,FeatureType.SIFT))] = log_reg_sift

feature_type = FeatureType.SURF.name
log_reg_file_path = os.path.join(model_outputs_dir,'logistic_regression_{}_training.pkl'.format(feature_type.lower()))
log_reg_surf= load_logistic_regression(feature_type,log_reg_file_path,model_outputs_dir,X_train,y_train,X_test,y_test)

feature_classifier_map[str(FeatureClassifierPair(ClassifierType.LOG_REGRESSION,FeatureType.SURF))] = log_reg_surf


feature_type = FeatureType.ORB.name
log_reg_file_path = os.path.join(model_outputs_dir,'logistic_regression_{}_training.pkl'.format(feature_type.lower()))
log_reg_orb= load_logistic_regression(feature_type,log_reg_file_path,model_outputs_dir,X_train,y_train,X_test,y_test)

feature_classifier_map[str(FeatureClassifierPair(ClassifierType.LOG_REGRESSION,FeatureType.ORB))] = log_reg_orb

"""###################################### MLP Loading #####################################"""


feature_type = FeatureType.SIFT.name
mlp_file_path = os.path.join(model_outputs_dir,'mlp_model_{}_training.pkl'.format(feature_type.lower()))

mlp_sift = load_mlp_model2(mlp_file_path,feature_type,X_train, y_train, X_test, y_test)
#mlp_sift = load_mlp_model(mlp_file_path, feature_type, X_train, y_train, X_test, y_test)
feature_classifier_map[str(FeatureClassifierPair(ClassifierType.MLP,FeatureType.SIFT))] = mlp_sift

feature_type = FeatureType.SURF.name
mlp_file_path = os.path.join(model_outputs_dir,'mlp_model_{}_training.pkl'.format(feature_type.lower()))
mlp_surf = load_mlp_model2(mlp_file_path,feature_type,X_train, y_train, X_test, y_test)
#mlp_surf = load_mlp_model(mlp_file_path, feature_type, X_train, y_train, X_test, y_test)
feature_classifier_map[str(FeatureClassifierPair(ClassifierType.MLP,FeatureType.SURF))] = mlp_surf

feature_type = FeatureType.ORB.name
mlp_file_path = os.path.join(model_outputs_dir,'mlp_model_{}_training.pkl'.format(feature_type.lower()))
mlp_orb = load_mlp_model2(mlp_file_path,feature_type,X_train, y_train, X_test, y_test)
#mlp_orb = load_mlp_model(mlp_file_path, feature_type, X_train, y_train, X_test, y_test)
feature_classifier_map[str(FeatureClassifierPair(ClassifierType.MLP,FeatureType.ORB))] = mlp_orb

"""###################################### Loading CNN Models vgg16 and and Resnet50 #####################################"""


labelencoder = LabelEncoder()

##label encode ytrainn and ytest
y_train_encoded = labelencoder.fit_transform(y_train)
y_test_encoded = labelencoder.transform(y_test)

# Create dataset from several tensors with matching first dimension
# Samples will be drawn from the first dimension (rows)
dataset = TensorDataset( Tensor(X_train), Tensor(y_train_encoded))

# Create a data loader from the dataset
# Type of sampling and batch size are specified at this step
train_loader = DataLoader(dataset, batch_size= 20)

test_dataset = TensorDataset( Tensor(X_test), Tensor(y_test_encoded) )

# Create a data loader from the dataset
# Type of sampling and batch size are specified at this step
test_loader = DataLoader(test_dataset, batch_size= 20)
n_classes = len(set(y_train))

del X_train,X_test
cnn_type = ClassifierType.CNN_ResNet
results_files= os.path.join(model_outputs_dir,'{}_results.pkl'.format(cnn_type.name))

resnet = load_cnn(os.path.join(model_outputs_dir,'resnet.pt'),results_files,ClassifierType.CNN_ResNet,n_classes,train_loader,test_loader)

feature_classifier_map[str(FeatureClassifierPair(ClassifierType.CNN_ResNet,None))] = resnet
resnet.plots(labelencoder.classes_,labelencoder.classes_)
torch.cuda.empty_cache()  
 
cnn_type = ClassifierType.CNN_ResNet 
vgg16 = load_cnn(os.path.join(model_outputs_dir,'vgg16.pt'),results_files,ClassifierType.CNN_vgg16,n_classes,train_loader,test_loader)
vgg16.plots(labelencoder.classes_,labelencoder.classes_)
feature_classifier_map[str(FeatureClassifierPair(ClassifierType.CNN_vgg16,None))] = vgg16

"""#####################################################################################################################

############################################### RECOGNISE FACE FUNC #################################################"""
""" recogniseFace by default feature_type is None, classifier_type = ClassifierType.CNN_ResNet, 
and creative_mode is None """
def recogniseFace(img_path, feature_extractor = None, classifier_type = ClassifierType.CNN_ResNet,creative_mode = None):
    print('Recognising Face')
    img = convertToRGB(cv2.imread(img_path))
    model = None
    try:
        model = feature_classifier_map[str(FeatureClassifierPair(classifier_type,feature_extractor))]
    except:
        raise ValueError("No Model Availble for classifier {} and feature type {}".format(classifier_type.name,feature_extractor))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     
    # Create face detector
    # thresholds: MTCNN face detection thresholds
    #Margin: to add to bounding box
    mtcnn = MTCNN(keep_all=True, device=device,thresholds=[0.8, 0.9, 0.9],margin=20 )

    frames = []
    persons = []
    
    # Detect face
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
    
    if(boxes is None or len(boxes) == 0):
        print('No faces detected')
        return persons
    
    #apply creative mode
    if creative_mode == CreativeModeType.cartoonify:
        img = cartoonify(img);
        
    fig, ax = plt.subplots(figsize=(30, 20))
    ##ax.imshow(img)
    ax.axis('off')

    for box, landmark in zip(boxes, landmarks):
            x1,y1,x2,y2 = box
            w,h = x2-x1,y2-y1
            try:
                face = cv2.resize(img[int(y1):int(y2), int(x1):int(x2)],(224,224))
            except Exception as e:
                print(str(e))
            #predict labe using model
            pred = model.predict(face)

            #add ID of face and centre point
            centre_x,centre_y = x1+w/2,y1+h/2
            
            if classifier_type == ClassifierType.CNN_ResNet or classifier_type == ClassifierType.CNN_vgg16: 
                pred = labelencoder.inverse_transform([pred])[0]
                
            persons.append((pred,(centre_x,centre_y)))

            frames.append(face)
            
            ##add bounding box around faces
            cv2.rectangle(img,  (x1,y1), (x2,y2),(0, 0, 255), 9)
            
            
            ##add prediction on top of persons head
            text = "{}".format(pred)
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10

            cv2.putText(img, text, (int(x1+30), int(y)),
    			cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 10)
            
            #plotting landmarks i.e. eyes,nose
            ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]), s=8)
            ax.scatter(landmark[:, 0], landmark[:, 1], s=6)
            

            if creative_mode == CreativeModeType.sunglasses_cigar:
                specs_ori = cv2.imread(os.path.join(createive_mode_asssts_dir,'sunglasses.png'),-1)
                cigar_ori = cv2.imread(os.path.join(createive_mode_asssts_dir,'cigar.png'),-1)
                img = add_glasses(img,x1,y1,w,h,specs_ori,cigar_ori)

    plt.imshow(img)
    persons = np.asanyarray(persons)
    print(persons)
    return persons
