# -*- coding: utf-8 -*-
"""
Created on Mon May 11 19:35:53 2020

@author: adds0
"""
from PIL import Image
import numpy as np
from classifiers import BaseClassifier
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from torchvision import transforms, models
from torch.autograd import Variable

class NeuralNet(nn.Module):
    def __init__(self, number_hidden_nodes, number_hidden_layers, activation,number_of_classes,image_width,color_channels, drop_out=0.5):
        super(NeuralNet, self).__init__()
        self.image_width = image_width
        self.number_hidden_layers = number_hidden_layers
        self.color_channels = color_channels
        self.number_hidden_nodes = number_hidden_nodes

        if activation == "sigmoid":
            self.activation_func = nn.Sigmoid()
        elif activation == "relu":
            self.activation_func = F.relu
            
    
        self.drop_out = drop_out
        # Set up perceptron layers and add dropout
        self.fc1 = nn.Linear(int(self.image_width * self.image_width * self.color_channels),
                                   self.number_hidden_nodes)
        
        # dropout prevents overfitting of data
        self.fc1_drop = nn.Dropout(drop_out)
        if number_hidden_layers == 2:
            self.fc2 = nn.Linear(self.number_hidden_nodes,
                                       self.number_hidden_nodes)
            # dropout prevents overfitting of data
            self.fc2_drop = nn.Dropout(drop_out)

        self.out = nn.Linear(self.number_hidden_nodes, number_of_classes)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, int(self.image_width * self.image_width * self.color_channels))
        #x = x.view(-1, int( x.shape[0] * x.shape[1]))
        
        # add hidden layer, with relu activation function
        x = self.activation_func (self.fc1(x))
        
        # add dropout layer
        x = self.fc1_drop(x)
        
        if self.number_hidden_layers == 2:
            # add hidden layer, with relu activation function
            x = self.activation_func (self.fc2(x))
            
             # add dropout layer
            x = self.fc2_drop(x)
            
        # add output layer     
        return F.log_softmax(self.out(x))
        #return F.softmax(self.out(x))



class MLPClassifier(BaseClassifier):
    def __init__(self,feature_type,number_hidden_nodes,number_hidden_layers):
        super().__init__(feature_type)
        self.number_hidden_layers = number_hidden_layers
        self.number_hidden_nodes = number_hidden_nodes
        self.k = None
        #self.feature_type = feature_type
        #self.features = None  
        self.kmeans = None
        self.histo_list  =None
        self.extract_feature_func = None
        self.model = None
        self.n_classes = None
        self.image_width = None
        self.color_channels = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    def train(self,x_train,y_train,activation,keep_rate,learning_rate= 0.001,num_epochs=10,batch_size = 20):
        self.color_channels = x_train.shape[3]
        
        if(self.feature_type is None or self.feature_type == 'None'):
            self.image_width = x_train.shape[1]
            self.labels_train = y_train.astype(int)
            self.x_train = x_train
        else:
            self.x_train,self.labels_train = self.extract_features_create_histogram(x_train,y_train.astype(int))
            self.x_train = np.asarray(self.x_train)
            self.image_width = math.sqrt(self.x_train.shape[1])
            self.color_channels = 1
            

        self.n_classes = len(set(y_train.astype(int)))
        
        if self.model is None:
            self.model = NeuralNet(self.number_hidden_nodes, self.number_hidden_layers, 
                                   activation,self.n_classes,self.image_width,self.color_channels, 1-keep_rate)
        self.model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)   
        
        #dataset = TensorDataset( Tensor(x_train), Tensor(y_train.astype(int)))
        dataset = TensorDataset( Tensor(self.x_train), Tensor(self.labels_train))
        self.train_loader = DataLoader(dataset, batch_size= batch_size)
        
        # Train the model
        total_step = len(self.train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):  
                # Move tensors to the configured device
                
                images = self.reshape(images).to(self.device)
                #images = images.permute(0,3,1,2).to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                labels = labels.long()
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    def reshape(self,images):
        if self.feature_type is None or self.feature_type.lower() == 'None':
            return images.permute(0,3,1,2)
        else:
            return images.reshape(-1, int(self.image_width*self.image_width))
        
    def save_models(self):
        torch.save(self.model.state_dict(), "mlp_model_{}.pt".format(self.feature_type))
        print("Checkpoint saved")
        
    def load_models(self,file_path):
        self.model.load_state_dict(torch.load(file_path))
    
        print("Model Loaded")

    def test(self,x_test,y_test,batch_size = 20):
        print('testing the model')
        if(self.feature_type is None or self.feature_type == 'None'):
            labels_test = y_test.astype(int)
            x_test = x_test
        else:
            x_test,labels_test = self.create_histograms(x_test,y_test,self.kmeans,self.k)
            x_test = np.asarray(x_test)

        dataset = TensorDataset( Tensor(x_test), Tensor(labels_test))
        test_loader = DataLoader(dataset, batch_size= batch_size)      
        # Test the model
        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = self.reshape(images).to(self.device)
                #images = images.permute(0,3,1,2).to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                labels = labels.long()
                print(labels)
                labels = labels-1
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                print("predicted:")
                print( predicted)
                correct += (predicted == labels).sum().item()
        
            print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
            
    def predict(self,image, image_resize = 224):
        if(isinstance(image,(np.ndarray, np.generic))):
            image = Image.fromarray(image)
        image_tensor = transforms.Compose([transforms.Resize(image_resize),transforms.ToTensor(),])(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(self.device)
        output = self.model(input)
        index = output.data.cpu().numpy().argmax()
