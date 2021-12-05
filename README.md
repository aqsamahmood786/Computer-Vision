# Computer-Vision
# Facial Recognition
### Author
Name: Aqsa mahmood

### Hardware Specification
This project was run on the following hardware specification:
| Hardware | Description |
| ------ | ------ |
| System Model:  | MS-7B93 |
| System Type | x64-based PC |
| Processor(s): |  AMD64 Family 23 Model 113 Stepping 0 AuthenticAMD ~3800 Mhz |
| Graphics | NVIDIA GeForce RTX 2070 SUPER |
| BIOS Version: | American Megatrends Inc. 1.00, 14/06/2019 |
|Total Physical Memory: |  16,332 MB|
### Code Structure
```
FacialRecognition
│
└───resources
│   │
│   └───model_outputs
│       │   CNN_ResNet_results.pkl
│       │   CNN_vgg16_results.pkl
│       │   logistic_regression_orb_training.pkl
│       │   logistic_regression_sift_training.pkl
│       │   logistic_regression_surf_training.pkl
│       │   mlp_model_orb_training.pkl
│       │   mlp_model_sift_training.pkl
│       │   mlp_model_surf_training.pkl
│       │   orb_histogram.pkl
│       │   resnet.pt
│       │   sift_histogram.pkl
│       │   surf_histogram.pkl
│       │   svm_orb_training.pkl
│       │   svm_sift_training.pkl
│       │   svm_surf_training.pkl
│       │   vgg16.pt
│   └───creative_mode_assets
│       │   cigar.png
│       │   mustache.png
│       │   sunglasses.png
│   
│    
└───scripts
│   │   classifiers.py
│   │   dataset_loader.py
│   │   face_extractor.py
│   │   loaders.py
│   │   mlp_classifier_pytorch.py
│   │   run_face_recognition.py
│   │   training.py
│   │   untilities.py
```
### Python File Description
 - loaders.py - this file contains methods/functions to load or train the machice learning and CNN models
 - classifiers.py - this file contains classes for SVMClassifier2,MLPClassifier2,LogisticRegressionClassifier, and CNNClassifier. This file also contains a class called BaseClassifier, which contains functionalty such as creating bag of word features, sift feautures....
 - run_face_recognition.py - this file essentially uses MTCNN to extarct faces from images and videos and saves them to directory
 - training.py - this file essentially loads the dataset of faces, and trains models and saves them to directory
 - untilities.py - this file contains general utility functions used throught the files
 - dataset_loader.py - loads the faces datasets
 - mlp_classifier_pytorch.py - this file contains a neural network defined in pytorch, however this is not used as it has been replaced by the MLP classifier provided by scikit-learn. The contents of this can be ignored.
### Model Downloads
The 'model_outputs' directory located within the resourcs folder contains trained models saved in the form of a pickle files (i.e. *.pkl files) or pytorch files which have an extention '*.pt' The feature extractors SIFT,SURF and ORB were applied to SVM,MLP and Logistic Regression classifier,thus given a total of 9 feature-classifier models plus two pretrained CNN models,namely vgg16 and resnet-50. This gives a total of 11 trained models for face recognition.

### Installation
First, install [Python 3.7](https://www.python.org/downloads/) and [Anconda](https://docs.anaconda.com/anaconda/install/)
Then we need to setup the anaconda enviroment. 
On a linux environment, navigate to the FacialRecognition project, run the follwing on command line as super user:
```sh
$ ./setup.sh
```
This script shouuld create the conda enviroment, with all devlopment dependencies, and activate the environment 'cvCoursework2020'
| Depenedency | Version |
| ------ | ------ |
| python  | 3.7.6 |
| pytorch | 1.2.0 |
| pillow |  6.1|
| torchvision | 0.4.0 |
| fastai | 1.0.57 |
| ipykernel: | 4.6.1 |
| pytest | 3.6.4 |
| bqplot |
| scikit-learn | 0.19.1 |
| pip | 19.0.3 |
| cython | 0.29.1 v
| papermill | 1.2.0 |
| black | 18.6b4 v
| ipywebrtc |
| lxml | 4.3.2 |
| pre-commit | 1.14.4 |
| pyyaml | 5.1.2 |
| requests | 2.22.0 v
| einops | 0.1.0 |
| cytoolz |
| seaborn |
| mtcnn |
| scikit-image |
| decord | 0.3.5 |
| nvidia-ml-py3 |
| nteract-scrapbook |
| azureml-sdk[notebooks,contrib] | 1.0.30 |
| facenet-pytorch |
| opencv-python | 3.4.2.16 |
| opencv-contrib-python | 3.4.2.16 |
| jupyter | 1.0.0 |
| spyder | 4.1.3 |

On a windows environment, can run the following script via Anaconda Prompt :
```sh
$ setup.bat
```
Note that setuping up the environment does take some time. Once, the the script has finished, the environment 'cvCoursework2020'. On the same Anaconda Prompt, spyder can be launched via the follwing command:

```sh
$ spyder
```


### Running Application

The python file of interest is **run_face_recognition.py**, which contains the recogniseFace function, and takes in the follwing parameters:
 - image file path
 - feature extractor type - this is defined by enum called FeatureType
  - classifier_typer type - this is defined by enum called ClassifierType
  - creative mode type - this alos defined by an enum called CreativeModeType

The signature of the function is defined below, and by deafult yiu can observe that the creative_mode is set to None:
```
def recogniseFace(img_path, feature_extractor = None, classifier_type = ClassifierType.CNN_ResNet,creative_mode = None):
```

**Note the enums FeatureType,ClassifierType,CreativeModeType are defined in the class run_face_recognition.py**
The recogniseFace function be invoked in the follwing ways (note <file_path> is the file path of an image):
```
  recogniseFace(<file_path>,None,ClassifierType.CNN_ResNet)
  recogniseFace(<file_path>,None,ClassifierType.CNN_ResNet,CreativeModeType.sunglasses_cigar)
```
The above line uses the Resnet-50 classifier, with creative mode set to sunglasses_cigar.With this creative mode faces should be overlayed with sunglasses and cigars. The line of code below allows you invoke the recogniseface with the vgg16 classifier without creative mode. 
```
  recogniseFace(<file_path>,None,ClassifierType.CNN_vgg16)
  recogniseFace(<file_path>,None,ClassifierType.CNN_vgg16,CreativeModeType.cartoonify)
```
Below we can see how to invoke the function using the SVM classifier but with the differenct feature extractors SIFT,SURF, and ORB. Furthemore we can also observe the use of the creative mode type cartoonify.
```
  recogniseFace(<file_path>,FeatureType.SURF,ClassifierType.SVM,CreativeModeType.cartoonify)
  recogniseFace(<file_path>,FeatureType.SURF,ClassifierType.SVM,CreativeModeType.sunglasses_cigar)
  recogniseFace(<file_path>,FeatureType.SURF,ClassifierType.SVM)
  recogniseFace(<file_path>,FeatureType.SIFT,ClassifierType.SVM,CreativeModeType.sunglasses_cigar)
  recogniseFace(<file_path>,FeatureType.SIFT,ClassifierType.SVM,CreativeModeType.cartoonify)
  recogniseFace(<file_path>,FeatureType.ORB,ClassifierType.SVM,CreativeModeType.sunglasses_cigar)
  recogniseFace(<file_path>,FeatureType.ORB,ClassifierType.SVM)
```

Below we can see how to invoke the function using the MLP classifier but with the differenct feature extractors SIFT,SURF, and ORB. Furthemore we can also observe the use of the creative mode type cartoonify.
```
  recogniseFace(<file_path>,FeatureType.SURF,ClassifierType.MLP,CreativeModeType.cartoonify)
  recogniseFace(<file_path>,FeatureType.SURF,ClassifierType.MLP,CreativeModeType.sunglasses_cigar)
  recogniseFace(<file_path>,FeatureType.SURF,ClassifierType.MLP)
  recogniseFace(<file_path>,FeatureType.SIFT,ClassifierType.MLP,CreativeModeType.sunglasses_cigar)
  recogniseFace(<file_path>,FeatureType.SIFT,ClassifierType.MLP,CreativeModeType.cartoonify)
  recogniseFace(<file_path>,FeatureType.ORB,ClassifierType.MLP,CreativeModeType.sunglasses_cigar)
  recogniseFace(<file_path>,FeatureType.ORB,ClassifierType.MLP)
```

Below we can see how to invoke the function using the Logistic Regression classifier but with the different feature extractors SIFT,SURF, and ORB. Furthemore we can also observe the use of the creative mode types cartoonify and sunglasses with a cigar.
```
  recogniseFace(<file_path>,FeatureType.SURF,ClassifierType.LOG_REGRESSION,CreativeModeType.cartoonify)
  recogniseFace(<file_path>,FeatureType.SURF,ClassifierType.LOG_REGRESSION,CreativeModeType.sunglasses_cigar)
  recogniseFace(<file_path>,FeatureType.SURF,ClassifierType.LOG_REGRESSION)
  recogniseFace(<file_path>,FeatureType.SIFT,ClassifierType.LOG_REGRESSION,CreativeModeType.sunglasses_cigar)
  recogniseFace(<file_path>,FeatureType.SIFT,ClassifierType.LOG_REGRESSION,CreativeModeType.cartoonify)
  recogniseFace(<file_path>,FeatureType.ORB,ClassifierType.LOG_REGRESSION,CreativeModeType.sunglasses_cigar)
  recogniseFace(<file_path>,FeatureType.ORB,ClassifierType.LOG_REGRESSION)
```



