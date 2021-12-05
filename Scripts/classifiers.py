# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 23:37:28 2020

@author: adds0
"""
from untilities import surf_features
from untilities import sift_features
from untilities import orb_features
from untilities import extract_sift
from untilities import extract_surf
from untilities import extract_orb
from untilities import gridSearch
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import os
import pickle
import torch.nn as nn
import bz2
import cv2
from untilities import write_to_pickle_bz2file
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt        
import seaborn as sns
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from torchvision import transforms, models
from torch.autograd import Variable

class BaseClassifier:
    def __init__(self, feature_type):
        self.feature_type = feature_type
        self.features = None
        self.k = None
        self.kmeans = None
        self.histo_list  =None
        self.labels_train = None
        self.classifier = None
    """
    To create our each image by a histogram. We will create a vector of k value for each image. 
    For each keypoints in an image, we will find the nearest center and increase by one its value
    
    """
    def create_histograms(self,x_train,y_train,kmeans,k):
        # Creates histograms for train data   
        histo_list = []
        labels_train = []
        for i in range(x_train.shape[0]):
            kp, des = self.extract_feature_func(x_train[i])
        
            if des is None:
                continue
            histo = np.zeros(k)
            nkp = np.size(kp)
            for d in des:
                idx = kmeans.predict([d])
                histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly
            histo_list.append(histo)
            labels_train.append(y_train[i])
        return histo_list,labels_train
    
    def extract_features(self,X_train,y_train):
        if(self.features is None):
            if(self.feature_type.lower() == "sift"):
                self.features = sift_features(X_train, y_train)
                self.extract_feature_func = extract_sift
            elif(self.feature_type.lower() == "surf"):
                self.features = surf_features(X_train, y_train)
                self.extract_feature_func = extract_surf
            elif(self.feature_type.lower() == "orb"):
                self.features = orb_features(X_train, y_train)
                self.extract_feature_func = extract_orb
            else:
                self.features = X_train,y_train
        return self.features
    
    def extract_features_function(self):
        extract_feature_func = None
        if(self.feature_type.lower() == "sift"):
            extract_feature_func = extract_sift
        elif(self.feature_type.lower() == "surf"):
            extract_feature_func = extract_surf
        elif(self.feature_type.lower() == "orb"):
            extract_feature_func = extract_orb
        return extract_feature_func
    
    def extract_features_create_histogram(self,x_train,y_train,verbose=0):
        self.extract_features(x_train,y_train)
        descriptor_list = self.features[0]
    
        """
        We now have an array with a huge number of descriptors. We cannot use all of
         them to create or model so we need to cluster them. A rule-of-thumb is to 
         create k centers with k = number of categories * 10 (in our case, it's 480)
        """
        self.k = len(set(y_train)) * 10
        
        #batch_size = np.size(os.listdir(img_path)) * 3
        self.kmeans = MiniBatchKMeans(n_clusters=self.k, batch_size=self.k*3, verbose=verbose).fit(descriptor_list)
        #visual_words  = kmeans.cluster_centers_
        print('Creating Histogram')
        return self.create_histograms(x_train,y_train,self.kmeans,self.k)    
    
    def get_histogram(self,x_train,y_train,feature_type=None):
        if(feature_type is None):
            feature_type = self.feature_type
        histo_list,labels_train,kmeans,k = None,None,None,None
        histogram_file = os.path.join(self.model_outputs_dir,'{}_histogram.pkl'.format(feature_type.lower()))
        if (not os.path.exists(histogram_file)):
            histo_list,labels_train = self.extract_features_create_histogram(x_train,y_train)
            write_to_pickle_bz2file(histogram_file,[histo_list,labels_train,self.kmeans,self.k])
        else: 
            with bz2.BZ2File(histogram_file, 'rb') as f:
                histo_list = pickle.load(f)
                labels_train = pickle.load(f)
                self.kmeans = pickle.load(f)
                self.k = pickle.load(f)
                f.close()
        return histo_list,labels_train
    
    def plot_confusion_matrix(self,x_test,y_test,class_names):
        # Plot non-normalized confusion matrix
        titles_options = [("Confusion matrix, without normalization", None),
                          ("Normalized confusion matrix", 'true')]
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(self.classifier, x_test, y_test,
                                         display_labels=class_names,
                                         cmap=plt.cm.Blues,
                                         normalize=normalize)
            disp.ax_.set_title(title)
        
            print(title)
            print(disp.confusion_matrix)
        
        plt.show()        

class SVMClassifier2(BaseClassifier):
    def __init__(self,feature_type):
        super().__init__(feature_type)
        self.extract_feature_func = self.extract_features_function()
        self.classifier = None
    
    def perform_hyperparameter_tuning(self,x_train,y_train,kernels, cs,gammas,cv = 10 ):
        print("performing hyperparameter tuning using grid search with 10 fold CV")
        if self.histo_list is None and self.labels_train is None:
            self.histo_list,self.labels_train = self.get_histogram(x_train,y_train)
        
        best = gridSearch(SVC(),cs,gammas,kernels,cv,np.array(self.histo_list),self.labels_train)
        print("finished hyperparameter tuning")
        self.c = best[0]['C']
        self.gamma = best[0]['gamma']
        self.kernel = best[0]['kernel']
        #self.classifier = SVC(kernel=self.kernel,C = self.c,gamma = self.gamma )
        return best
        
    def classify(self,x_train,y_train,x_test,y_test,kernel  , c = None  , gamma = None  ):
        
        if self.histo_list is None and self.labels_train is None:
            self.histo_list,self.labels_train = self.get_histogram(x_train,y_train)
                
        if(c is None and gamma is None and self.classifier is None ):
            self.classifier = SVC(kernel=self.kernel,C = self.c,gamma = self.gamma)
        else:
            #svm = SVMClassifier("rbf", 10, 0.00001, np.array(histo_list), labels_train)
            self.classifier = SVC(kernel=kernel,C = c,gamma = gamma)
        print ("Training the classifyer ...")
        self.classifier.fit(np.array(self.histo_list), self.labels_train)
        print ("Validating the classifyer ...")
        self.test(self.classifier,x_test,y_test,self.k,self.kmeans)
        
    def test(self,svm,x_test,y_test,k,kmeans):
        self.accuracy = 0
        total_imgs = len(x_test)
        
        for i in range( x_test.shape[0]):
            #im = cv2.resize(x_test[i].copy(),(80,80))
            im = x_test[i]
            kp, des = self.extract_feature_func(im)
        
            x = np.zeros(k)
            nkp = np.size(kp)
        
            if des is None:
                continue
            for d in des:
                idx = kmeans.predict([d])
                x[idx] += 1/nkp    
            
            
            #pred = svm.predict(des)
            pred = svm.predict([x])
            pred = list(map(lambda x: int(x),pred))
   
            real_label = y_test[i]
        
            counts = np.bincount(pred)
            pred_label = np.argmax(counts)
            if int(real_label) == pred_label :
                self.accuracy +=1
            #print img[0] + " - Real labe: " + str(real_label) + " - Pred label:" + str(pred_label)
        self.accuracy = self.accuracy/total_imgs*100
        print ("Accuracy obtained using {} is: {}%".format(self.feature_type,self.accuracy))  
        
    def predict(self,img):
        kp, des = self.extract_feature_func(img)
        
        x = np.zeros(self.k)
        nkp = np.size(kp)
    
        if des is None:
            return
        for d in des:
            idx = self.kmeans.predict([d])
            x[idx] += 1/nkp    
        
        
        #pred = svm.predict(des)
        pred = self.classifier.predict([x])
        pred = list(map(lambda x: int(x),pred))
        counts = np.bincount(pred)
        pred_label = np.argmax(counts)
        return pred_label
    
class MLPClassifier2(BaseClassifier):
    def __init__(self,feature_type):
        super().__init__(feature_type)
        self.extract_feature_func = self.extract_features_function()
        self.classifier = None
    
        
    def classify(self,x_train,y_train,x_test,y_test,hidden_layer_sizes=(150,100,50),
                                                max_iter=300,activation = 'relu',solver='adam',random_state=1  ):
        
        if self.histo_list is None and self.labels_train is None:
            self.histo_list,self.labels_train = self.get_histogram(x_train,y_train)
                
        if self.classifier is None:
            self.classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                                max_iter=max_iter,activation = activation,solver=solver,random_state=random_state)
        print ("Training the classifyer ...")
        self.classifier.fit(np.array(self.histo_list), self.labels_train)
        print ("Validating the classifyer ...")
        self.test(self.classifier,x_test,y_test,self.k,self.kmeans)
        
    def test(self,classifier,x_test,y_test,k = None,kmeans = None):
        if k is None:
            k = self.k
        if kmeans is None:
            kmeans =self.kmeans
        self.accuracy = 0
        total_imgs = len(x_test)
        
        for i in range( x_test.shape[0]):
            #im = cv2.resize(x_test[i].copy(),(80,80))
            im = x_test[i]
            kp, des = self.extract_feature_func(im)
        
            x = np.zeros(k)
            nkp = np.size(kp)
        
            if des is None:
                continue
            for d in des:
                idx = kmeans.predict([d])
                x[idx] += 1/nkp    
            
            
            #pred = svm.predict(des)
            pred = classifier.predict([x])
            pred = list(map(lambda x: int(x),pred))   
            real_label = y_test[i]
        
            counts = np.bincount(pred)
            pred_label = np.argmax(counts)
            if int(real_label) == pred_label :
                self.accuracy +=1
            #print img[0] + " - Real labe: " + str(real_label) + " - Pred label:" + str(pred_label)
        self.accuracy = self.accuracy/total_imgs*100
        print ("Accuracy obtained using {} is: {}%".format(self.feature_type,self.accuracy))  
        
    def predict(self,img):
        kp, des = self.extract_feature_func(img)
        
        x = np.zeros(self.k)
        nkp = np.size(kp)
    
        if des is None:
            return
        for d in des:
            idx = self.kmeans.predict([d])
            x[idx] += 1/nkp    
        
        
        #pred = svm.predict(des)
        pred = self.classifier.predict([x])
        pred = list(map(lambda x: int(x),pred))
        counts = np.bincount(pred)
        pred_label = np.argmax(counts)
        return pred_label

class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self, feature_type):
        super().__init__(feature_type)
        self.extract_feature_func = self.extract_features_function()
        self.classifier = None
    
        
    def classify(self,x_train,y_train,x_test,y_test,c=1.0 ):
        
        if self.histo_list is None and self.labels_train is None:
            self.histo_list,self.labels_train = self.get_histogram(x_train,y_train)
                
        if self.classifier is None:
            self.classifier = LogisticRegression(C =c)
        print ("Training the classifyer ...")
        self.classifier.fit(np.array(self.histo_list), self.labels_train)
        print ("Validating the classifyer ...")
        self.test(self.classifier,x_test,y_test,self.k,self.kmeans)
        
    def test(self,classifier,x_test,y_test,k = None,kmeans = None):
        if k is None:
            k = self.k
        if kmeans is None:
            kmeans =self.kmeans
        self.accuracy = 0
        total_imgs = len(x_test)
        
        for i in range( x_test.shape[0]):
            #im = cv2.resize(x_test[i].copy(),(80,80))
            im = x_test[i]
            kp, des = self.extract_feature_func(im)
        
            x = np.zeros(k)
            nkp = np.size(kp)
        
            if des is None:
                continue
            for d in des:
                idx = kmeans.predict([d])
                x[idx] += 1/nkp    
            
            
            #pred = svm.predict(des)
            pred = classifier.predict([x])
            pred = list(map(lambda x: int(x),pred))   
            real_label = y_test[i]
        
            counts = np.bincount(pred)
            pred_label = np.argmax(counts)
            if int(real_label) == pred_label :
                self.accuracy +=1
            #print img[0] + " - Real labe: " + str(real_label) + " - Pred label:" + str(pred_label)
        self.accuracy = self.accuracy/total_imgs*100
        print ("Accuracy obtained using {} is: {}%".format(self.feature_type,self.accuracy))  
        
    def predict(self,img):
        kp, des = self.extract_feature_func(img)
        
        x = np.zeros(self.k)
        nkp = np.size(kp)
    
        if des is None:
            return
        for d in des:
            idx = self.kmeans.predict([d])
            x[idx] += 1/nkp    
        
        
        #pred = svm.predict(des)
        pred = self.classifier.predict([x])
        pred = list(map(lambda x: int(x),pred))
        counts = np.bincount(pred)
        pred_label = np.argmax(counts)
        return pred_label
    
class CNNClassifier():
    def __init__(self,number_of_classes,cnn_classifier_model = 'resnet50',freeze_cnn_layers = False):
        self.number_of_classes = number_of_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cnn_classifier_model = cnn_classifier_model
        if(cnn_classifier_model == 'CNN_vgg16'):
            print('using pretrained vgg16 cnn')
            self.model = models.vgg16(pretrained = True)
        else:
            print('using pretrained resnet50 cnn')
            self.model = models.resnet50(pretrained = True)
        
        for param in self.model.parameters():
            param.requires_grad = not freeze_cnn_layers
            
        if(cnn_classifier_model == 'CNN_vgg16'):
            self.model.classifier[6] = self.create_output_layer(self.model.classifier[6].in_features,0.4)
        else:
            #self.model.fc = nn.Linear(self.model.fc.in_features, number_of_classes)
            self.model.fc = self.create_output_layer(self.model.fc.in_features,0.4)
            
        self.model.to(self.device)
        self.labelencoder = LabelEncoder()
        
        
    def create_output_layer(self,number_inputs,dropout):
        return nn.Sequential(
                          nn.Linear(number_inputs, 256), 
                          nn.ReLU(), 
                          nn.Dropout(dropout),
                          nn.Linear(256, self.number_of_classes),                   
                          nn.LogSoftmax(dim=1))
    def train(self,train_loader,test_loader,learning_rate= 0.0001,num_epochs=10):

        #Error
        criterion = nn.CrossEntropyLoss()
        
        #Optimization algorithm
        optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)# lr: learning rate
        CUDA_LAUNCH_BLOCKING=1
        torch.backends.cudnn.enabled = False
        
        #Training of model
        self.running_loss_history = []
        self.running_corrects_history = []
        self.val_running_loss_history = []
        self.val_running_corrects_history = []

        # Train and vaidate the model the model
        total_step = len(train_loader)
        

        for epoch in range(num_epochs):
            running_loss = 0.0
            running_corrects = 0.0
      
            for i, (images, labels) in enumerate(train_loader):  
                # Move tensors to the configured device
                
                images = images.permute(0,3,1,2).to(self.device)
                #images = images.permute(0,3,1,2).to(self.device)
                labels = labels.to(self.device)
                #prepare model for training
                self.model.train()   
                # Forward pass
                outputs = self.model(images)
                labels = labels.long()
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, preds = torch.max(outputs, 1)
                loss_value = loss.item()
                running_loss += loss_value
                running_corrects += torch.sum(preds == labels.data)
                
                # defrag cached memory
                torch.cuda.empty_cache()
                
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Training Loss: {:.4f}' 
                           .format(epoch+1, num_epochs, i+1, total_step, loss_value)) 
            # After training loops ends, start validation
            else:
                
                val_correct,val_total,val_running_loss = self.test(test_loader)
                
                #accumulate test epoch loss
                self.val_running_loss_history.append(val_running_loss/len(test_loader.dataset))
                
                #accumulate test epoch accuracy
                self.val_running_corrects_history.append(val_correct/ len(test_loader.dataset))
   
                #accumulate training epoch loss
                self.running_loss_history.append(running_loss/len(train_loader.dataset))
            
                #accumulate training epoch accuracy
                self.running_corrects_history.append(running_corrects/ len(train_loader.dataset))

                print('Epoch :', (epoch+1))
                print('Training loss: {:.4f},   Training accuracy {:.4f} '
                      .format(self.running_loss_history[-1], self.running_corrects_history[-1].item()))
                print('Validation loss: {:.4f}, Validation accuracy {:.4f} '
                      .format(self.val_running_loss_history[-1], self.val_running_corrects_history[-1]))
  

    def test(self,test_loader,criterion = None):
        print('Testing the model')
        if criterion is None:
            #Error
            criterion = nn.CrossEntropyLoss()
                
        self.confusion_matrix = torch.zeros(self.number_of_classes, self.number_of_classes)

        correct = 0.0
        total = 0.0
        running_loss = 0.0
        # Test the model
        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():

            for images, labels in test_loader:
                images = images.permute(0,3,1,2).to(self.device)
                labels = labels.to(self.device)
                #prepare model for evaluation
                self.model.eval()        
                #forward pass
                outputs = self.model(images)
                labels = labels.long()
                
                # Calculate validation accuracy
                val_loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                running_loss += val_loss.item()
                correct += (predicted == labels).sum().item()
                
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    self.confusion_matrix[t.long(), p.long()] += 1
        
            #print('Accuracy of the network on the {} test images: {} %'.format(len(test_loader),100 * correct / total))
            
        return correct,total,running_loss
    
    def predict2(self,image, image_resize = 224):
        if(isinstance(image,(np.ndarray, np.generic))):
            image = Image.fromarray(image)
     
        test_transforms = transforms.Compose([transforms.Resize([224,224]),
                               transforms.ToTensor(),
                               ])  
         
        image_tensor = test_transforms(image).float()
        
        image_tensor = image_tensor.unsqueeze_(0)
        input_var = Variable(image_tensor)
        input_var = input_var.to(self.device)
        self.model.eval()
        output = self.model(input_var)
        
        #return index of highest probability
        index = output.data.cpu().numpy().argmax()
        return index  
    def predict(self,image):
        #modelX=torch.load('trained_models/facial_.pth')
        self.model.eval()
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        image_tensor = Tensor(cv2.resize(image,(224,224))).float()
        image_tensor = image_tensor.unsqueeze_(0)
        image_tensor = image_tensor.permute(0,3,1,2)
    
        input = Variable(image_tensor)
        input = input.to(device)
        output = self.model(input)
        index = output.data.cpu().numpy().argmax()
        return index    
    def plots(self,x_labels,y_labels):
        plt.plot(self.running_loss_history, label='training loss')
        plt.plot(self.val_running_loss_history, label='validation loss')
        plt.legend()
        
        plt.plot(self.running_corrects_history, label='training accuracy')
        plt.plot(self.val_running_corrects_history, label='validation accuracy')
        plt.legend()
        plt.figure(figsize=(100,100))  

        plt.figure(figsize=(15,15))
        ax= plt.subplot()
        print(self.confusion_matrix)
        sns_plot = sns.heatmap(self.confusion_matrix, annot=True, ax = ax,annot_kws={"size": 8},fmt=".1f"); #annot=True to annotate cells
        
        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(x_labels); 
        ax.yaxis.set_ticklabels(y_labels);
        
        sns_plot.figure.savefig("{}.png".format(self.cnn_classifier_model))
            
