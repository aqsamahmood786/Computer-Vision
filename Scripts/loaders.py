# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:56:33 2020

@author: adds0
"""

import os
import pickle
import bz2
import torch

from untilities import write_to_pickle_file
from untilities import write_to_pickle_bz2file

from classifiers import SVMClassifier2
from classifiers import LogisticRegressionClassifier
from classifiers import MLPClassifier2
from classifiers import CNNClassifier
from mlp_classifier_pytorch import MLPClassifier


def perform_tuning(x_train,y_train,directory,svm):
    c = None
    gamma = None
    kernel = None
    if(not os.path.exists(directory)):
        Cs = [0.001, 0.01, 0.1, 1, 10,100]
        gammas = [0.001, 0.01, 0.1, 1,10]
        kernels = ['rbf','linear','poly']
        best = svm.perform_hyperparameter_tuning(x_train,y_train,kernels,Cs,gammas)
        c = best[0]['C']
        gamma = best[0]['gamma']
        kernel = best[0]['kernel']
        write_to_pickle_file(directory,[best])
    else:
        f = open(directory, 'rb')
        best = pickle.load(f)
        c = best[0]['C']
        gamma = best[0]['gamma']
        kernel = best[0]['kernel']
        f.close()

    return c,gamma,kernel

def load_svm(feature_type,svm_model_file_path,model_outputs_dir,x_train,y_train,x_test,y_test,c=1,gamma=10,kernel ="rbf",hyperparameter_tuning = False):
    svm_classifier = SVMClassifier2(feature_type)
    if(not os.path.exists(svm_model_file_path)):
        svm_best = os.path.join(model_outputs_dir,'svm_{}_{}_training.pkl'.format(feature_type.lower(),'best'))

        if(hyperparameter_tuning):
           c,gamma,kernel = perform_tuning(x_train,y_train,svm_best,svm_classifier)
           
            
        svm_classifier.classify(x_train,y_train,x_test,y_test,kernel, c, gamma)
        write_to_pickle_bz2file(svm_model_file_path,[svm_classifier.classifier,svm_classifier.kmeans,svm_classifier.k,svm_classifier.accuracy])

    else:
        with bz2.BZ2File(svm_model_file_path, 'rb') as f:
            svm_classifier.classifier = pickle.load(f)
            svm_classifier.kmeans = pickle.load(f)
            svm_classifier.k = pickle.load(f)
            svm_classifier.accuracy = pickle.load(f)
            f.close()
    return svm_classifier


def load_logistic_regression(feature_type,model_file_path,model_outputs_dir,x_train,y_train,x_test,y_test,c=1.0):
    log_reg_classifier = LogisticRegressionClassifier(feature_type)
    if(not os.path.exists(model_file_path)):
        log_reg_classifier.classify(x_train,y_train,x_test,y_test)
        write_to_pickle_bz2file(model_file_path,[log_reg_classifier.classifier,log_reg_classifier.kmeans,log_reg_classifier.k,log_reg_classifier.accuracy])

    else:
        with bz2.BZ2File(model_file_path, 'rb') as f:
            log_reg_classifier.classifier = pickle.load(f)
            log_reg_classifier.kmeans = pickle.load(f)
            log_reg_classifier.k = pickle.load(f)
            log_reg_classifier.accuracy =pickle.load(f)
            f.close()
    return log_reg_classifier
def load_cnn(file_path,results_filec,cnn_type,number_classes,train_loader,test_loader,num_epochs=10,learning_rate= 0.0001,freeze_cnn_layers = False,map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):        
    cnn = CNNClassifier(number_classes,cnn_type.name,freeze_cnn_layers)
    #results_filec= os.path.join(model_outputs_dir,'{}_results.pkl'.format(cnn_type.name))
    if(not os.path.exists(file_path)):
        
        cnn.train(train_loader,test_loader,learning_rate,num_epochs)
        #cnn.plot()
        torch.save(cnn.model.state_dict(), file_path)
        write_to_pickle_bz2file(results_filec,[cnn.val_running_corrects_history,cnn.val_running_loss_history,
                                               cnn.running_corrects_history,cnn.running_loss_history,cnn.confusion_matrix])
        
    else:
        print('loading cnn {} classifier'.format(cnn_type.name))
        cnn.model.load_state_dict(torch.load(file_path,map_location=map_location))
        with bz2.BZ2File(results_filec, 'rb') as f:
             cnn.val_running_corrects_history = pickle.load(f)
             cnn.val_running_loss_history = pickle.load(f)
             cnn.running_corrects_history = pickle.load(f)
             cnn.running_loss_history =  pickle.load(f)
             cnn.confusion_matrix =  pickle.load(f)
             f.close()        

        
    return cnn
def load_mlp_model(file_path,feature_type,x_train,y_train,x_test,y_test):
    if(not os.path.exists(file_path)):
        mlpClass = MLPClassifier(feature_type,100,2)
        mlpClass.train(x_train,y_train,'relu',0.5)
        mlpClass.test(x_test,y_test,20)
        mlpClass.save_models()
    else:
        mlpClass = MLPClassifier(feature_type,100,2)
        mlpClass.load_models(file_path)
    return mlpClass

def load_mlp_model2(file_path,feature_type,x_train,y_train,x_test,y_test):
    if(not os.path.exists(file_path)):
        mlpClass = MLPClassifier2(feature_type)
        mlpClass.classify(x_train, y_train, x_test, y_test)
        
        write_to_pickle_bz2file(file_path,[mlpClass.classifier,mlpClass.kmeans,mlpClass.k,mlpClass.accuracy])
    else:
        mlpClass = MLPClassifier2(feature_type)
        with bz2.BZ2File(file_path, 'rb') as f:
            mlpClass.classifier = pickle.load(f)
            mlpClass.kmeans = pickle.load(f)
            mlpClass.k = pickle.load(f)
            mlpClass.accuracy =  pickle.load(f)
            f.close()
    return mlpClass