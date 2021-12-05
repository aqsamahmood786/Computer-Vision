# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 02:23:10 2020

@author: aqsa
"""
#import OpenCV module
import cv2
#import os module for reading training data directories and paths
import os
import PIL
from scipy.ndimage import filters
from skimage.util import random_noise
from skimage.transform import rotate
import numpy as np
import pickle
import bz2
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
from PIL import Image



def add_glasses(img,x1,y1,w,h,specs_original,cigar_orignal):
    glass_symin = int(y1 + 1.5 * h / 5)
    glass_symax = int(y1 + 2.5 * h / 5)
    sh_glass = glass_symax - glass_symin
    
    cigar_symin = int(y1 + 4 * h / 6)
    cigar_symax = int(y1 + 5 * h / 6)
    sh_cigar = cigar_symax - cigar_symin
 
    face_glass_roi_color = img[glass_symin:glass_symax, int(x1):int(x1+w)]
    face_cigar_roi_color = img[cigar_symin:cigar_symax, int(x1):int(x1+w)]

    specs = cv2.resize(specs_original, (w, sh_glass),interpolation=cv2.INTER_CUBIC)
    cigar= cv2.resize(cigar_orignal, (w, sh_cigar),interpolation=cv2.INTER_CUBIC) 
    
    img[glass_symin:glass_symax, int(x1):int(x1+w)] = transparentOverlay(face_glass_roi_color,specs)
    img[cigar_symin:cigar_symax, int(x1):int(x1+w)] = transparentOverlay(face_cigar_roi_color,cigar,x = int(w/2), y = int(sh_cigar/2))
    return img
    
def add_moustache(img):
    print('TODO')
    
def transparentOverlay(foreground, overlay, x =0, y=0, scale=1):
    foreground = foreground.copy()
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = foreground.shape  # Size of background Image

    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if y + i >= rows or x + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            foreground[y + i][x + j] = alpha * overlay[i][j][:3] + (1 - alpha) * foreground[y + i][x + j]
    return foreground

def watermarking(original, watermarked, alpha = 1, x=0, y=0):
  overlay = transparentOverlay(original, watermarked, x, y)
  output = original.copy()
  cv2.addWeighted(overlay, 1, output, 1 - 1, 0, output)
  return output

"""
This method was not copied, however guidance was taken from https://www.geeksforgeeks.org/cartooning-an-image-using-opencv-python/
"""
def cartoonify(image,num_downsampling_steps = 2,num_bilateral_filtering_steps = 7):
    
    # apply Gaussian pyramid in order to downsample the image
    image_color = image
    for i in range(num_downsampling_steps):
       image_color = cv2.pyrDown(image_color)
    
    # continously apply small bilateral filters instead of applying one large filter
    for i in range(num_bilateral_filtering_steps):
        image_color = cv2.bilateralFilter(image_color, d=9, sigmaColor=9, sigmaSpace=1)
    
    # upsample image to original size
    for i in range(num_downsampling_steps):
       image_color = cv2.pyrUp(image_color)
    
    #apply a median filter to reduce noise and convert to grayscale and apply median blur
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)
    
    #Use adaptive thresholding to create an edge mask
    # detect and enhance edges
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
       cv2.ADAPTIVE_THRESH_MEAN_C,
       cv2.THRESH_BINARY,
       blockSize=9,
       C=2)

    # Combine color image with edge mask & display picture
    # convert back to color, bit-AND with color image
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    return cv2.bitwise_and(image_color, img_edge)

def motion_blur(img,kernel = 'horizontal',kernel_size=10):
    # Specify the kernel size. 
    # The greater the size, the more the motion. 

    # Create the vertical kernel. 
    kernel_v = np.zeros((kernel_size, kernel_size)) 
      
    # Create a copy of the same for creating the horizontal kernel. 
    kernel_h = np.copy(kernel_v) 
    
    if(kernel == 'horizontal'):
        # Fill the middle row with ones.
        kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
        
        # Normalize. 
        kernel_h /= kernel_size 
        
        motion_blur = None
        # Apply the horizontal kernel. 
        motion_blur = cv2.filter2D(img, -1, kernel_h) 
    else:
        # Fill the middle row with ones. 
        kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
        # Normalize. 
        kernel_v /= kernel_size 
        
        # Apply the vertical kernel. 
        motion_blur = cv2.filter2D(img, -1, kernel_v) 
      
    
    return motion_blur

def rotate_image(img,angle=45):
  """
  apply rotation to an image  
  """
  return rotate(img,angle=angle)

def noise(img):
  """
  apply random noise to image  
  """
  return random_noise(img)

def gaussian_filter(img,sigma = 3):
  """
  applied gaussian blur to image 
  """
  return filters.gaussian_filter(img,sigma)
  
# Different data augmentation techniques
#Gaussian Blur
def gaussian_blur(img,kernel=(5,5)):
    """
    applied gaussian blur to image 
    """
    return cv2.GaussianBlur(img,kernel,0)

#Median Blur
def median_blur(img,kernel=5):
    """
    applied median blur to image 
    """
    return cv2.medianBlur(img,kernel) #median blur
#Expand Brightness of Image
def brightness(img,phi = 2, theta = 2, maxIntensity = 255.0):
    """
    expands brightness of image
    """
    return (maxIntensity/phi)*(img/(maxIntensity/theta))**0.5

# Reduce brightness of Image
def darker(img, phi = 2, theta = 2, maxIntensity = 255.0):
    """
    reduce brightness of image
    """
    return (maxIntensity/phi)*(img/(maxIntensity/theta))**2

def horizontal_flip(image_array):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

def extract_hog(img,  orientations=8, pixels_per_cell=(16,16),cells_per_block=(4, 4),block_norm= 'L2',visualise=True):    
    """
    Extract HOG keypoints and descriptors from an image
    
    """ 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    fd,hog_image = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,block_norm= block_norm,visualise=visualise)

    return hog_image,fd

def hog_features(x,y):
    descriptors = []
    labels = []
    #hog_image = {}
    print ('Applying HOG ...')
    for i in range(x.shape[0]):
        hog_img,fd = extract_hog(x[i])
        
        if fd is None:
            continue
     
        descriptors.append(fd)
        labels.append(y[i])
        #hog_image.setdefault(y[i],[]).append(fd)
    return [descriptors, labels]  

#Extract descriptors using Sift
 
def extract_sift(img, nfeatures = 0,octave=3, contrast=0.03, edge=10, sigma=1.6):    
    """
    Extract SIFT keypoints and descriptors from an image
    
    """ 
    if  isinstance(img,PIL.Image.Image):
        img = np.array(img)
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=octave, contrastThreshold=contrast, edgeThreshold=edge, sigma=sigma,nfeatures=nfeatures)
    kp, des = sift.detectAndCompute(gray,None)
    
    return kp,des


def sift_features(x,y):
    descriptors = []
    sift_vectors = {}
    print ('Applying Sift ...')
    for i in range(x.shape[0]):
        kp, des = extract_sift(x[i])
    
        if des is None:
            continue
     
        descriptors.extend(des)
        sift_vectors.setdefault(y[i],[]).append(des)
    return [descriptors, sift_vectors]   


# Extract descriptors using Surf
def extract_surf(img, hessian=50, octave=4, octaveLayers=2, ext=False,upright=False):    
    """
    Extract SURF keypoints and descriptors from an image
    """
    if  isinstance(img,PIL.Image.Image):
        img = np.array(img)
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold =hessian, nOctaves=octave, nOctaveLayers=octaveLayers, extended=ext,upright=upright)

    kp, des = surf.detectAndCompute(img,None)
    
    return kp,des

def surf_features(x,y):
    descriptors = []
    sift_vectors = {}
    print ('Applying Surf ...')
    for i in range(x.shape[0]):
        kp, des = extract_surf(x[i])
    
        if des is None:
            continue
     
        descriptors.extend(des)
        sift_vectors.setdefault(y[i],[]).append(des)
    return [descriptors, sift_vectors] 

def extract_orb(img, nfeatures = 500,scale_factor=1.2, nlevels=8, edge=31, first_level=0,wta_k = 2,patch_size = 31,fast_threshold = 20):    
    """
    Extract ORB keypoints and descriptors from an image
    int     nfeatures = 500,
    float     scaleFactor = 1.2f,
    int     nlevels = 8,
    int     edgeThreshold = 31,
    int     firstLevel = 0,
    int     WTA_K = 2,
    int     scoreType = ORB::HARRIS_SCORE,
    int     patchSize = 31,
    int     fastThreshold = 20 
    """ 
    if  isinstance(img,PIL.Image.Image):
        img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures,scaleFactor = scale_factor,
                          nlevels = nlevels,edgeThreshold = edge,
                          firstLevel = first_level,WTA_K=wta_k,
                          patchSize=patch_size,fastThreshold=fast_threshold)
    kp, des = orb.detectAndCompute(gray,None)
    
    return kp,des
def orb_features(x,y):
    descriptors = []
    sift_vectors = {}
    print ('Applying Orb ...')
    for i in range(x.shape[0]):
        kp, des = extract_orb(x[i])
    
        if des is None:
            continue
     
        descriptors.extend(des)
        sift_vectors.setdefault(y[i],[]).append(des)
    return [descriptors, sift_vectors] 


      
"""
   Input: Classifier, range of values for the parameters,  data and labels coresponding to the data
   output: Best parameters with the corresponding score
"""
def gridSearch(classifier,C_range, gamma_range,kernels,crossVal, data, labels):
    param_grid = dict(gamma=gamma_range, C=C_range,kernel =kernels )
    grid = GridSearchCV(classifier, param_grid=param_grid, cv=crossVal)
    grid.fit(data, labels)
    return grid.best_params_, grid.best_score_

def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def write_image_to_directory(file_name,directory,image):
    if not os.path.exists(directory):
        create_directory(directory)
    cv2.imwrite(os.path.join(directory,file_name), image.copy())
    
def create_directory(directory):
    if not os.path.exists(directory):
        try:
            print("New directory created {}".format(directory))
            os.makedirs(directory)
        except:
            print("Could not create {} directory".format(directory))
            
def write_to_pickle_file(file_path,objects):
    
    f = open(file_path, 'wb') 
    for obj in objects:
        pickle.dump(obj,f)  

    f.close()
    print("Finished writing {} to pickle file {}".format(objects,os.path.basename(file_path)))
    
def write_to_pickle_bz2file(file_path,objects):
    
    sfile = bz2.BZ2File(file_path, 'w')
    for obj in objects:
        pickle.dump(obj,sfile)  

    sfile.close()
    print("Finished writing {} to bz2 pickle file {}".format(objects,os.path.basename(file_path)))