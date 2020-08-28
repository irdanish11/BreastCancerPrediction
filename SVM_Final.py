# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 21:54:52 2020

@author: Danish
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display

import pandas as pd
import numpy as np

from PIL import Image

from skimage.feature import hog
from skimage.color import rgb2grey

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import roc_curve, auc
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import os
import glob as gb
import cv2
from tqdm import tqdm
import pickle
from sklearn.metrics import confusion_matrix
import seaborn 



# Creating Labels for Y_test and Y_trian
code = {'Benign':0 ,'Insitu':1 ,'InvGrade1':2, 'InvGrade2':3, 'InvGrade3': 4}

def create_features(img):
    # flatten three channel color image
    color_features = img.flatten()
    # convert image to greyscale
    grey_image = rgb2grey(img)
    # get HOG features from greyscale image
    hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    # combine color and hog features into a single array
    flat_features = np.hstack(color_features)
    return flat_features

def create_feature_matrix(X):
    features_list = []
    for i in tqdm(range(len(X))):
        # get features for image
        img = X[i]
        image_features = create_features(img)
        features_list.append(image_features)
        
    # convert list of arrays into a matrix
    feature_matrix = np.array(features_list)
    return feature_matrix

def plot_confusion_matrix(data, labels, output_filename):
    """Plot confusion matrix using heatmap.
     
    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.
     
    """
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))
    plt.title("Confusion Matrix")
    seaborn.set(font_scale=1.2)
    ax = seaborn.heatmap(data, annot=True, cmap=plt.cm.Blues, cbar_kws={'label': 'Scale'})
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set(ylabel="True Label", xlabel="Predicted Label")
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
def pickle_dump(obj, path):
    f = open(path,"wb")
    pickle.dump(svm,f)
    f.close()

def getcode(n) : 
    for x , y in code.items() : 
        if n == y : 
            return x

###################### Loading Data ######################
path = './Data/SVM'
##Transfering images to Array
def Dataset_loader(DIR, RESIZE, sigmaX=10):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR,IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".png":
            img = read(PATH)
           
            img = cv2.resize(img, (RESIZE,RESIZE))
           
            IMG.append(np.array(img))
    return IMG

benign_train = np.array(Dataset_loader(path+'/TrainingSet/Benign',100))
insitu_train = np.array(Dataset_loader(path+'/TrainingSet/Insitu',100))
invasive1_train = np.array(Dataset_loader(path+'/TrainingSet/InvGrade1',100))
invasive2_train = np.array(Dataset_loader(path+'/TrainingSet/InvGrade2',100))
invasive3_train = np.array(Dataset_loader(path+'/TrainingSet/InvGrade3',100))

benign_test = np.array(Dataset_loader(path+'/ValidationSet/Benign',100))
insitu_test = np.array(Dataset_loader(path+'/ValidationSet/Insitu',100))
invasive1_test = np.array(Dataset_loader(path+'/ValidationSet/InvGrade1',100))
invasive2_test = np.array(Dataset_loader(path+'/ValidationSet/InvGrade2',100))
invasive3_test = np.array(Dataset_loader(path+'/ValidationSet/InvGrade3',100))

##Creating Lables and Preparing Training and Testing Datasets

# Creating X_test and X_train 
X_train = np.concatenate((benign_train, insitu_train, invasive1_train, invasive2_train, invasive3_train), axis = 0)
X_test = np.concatenate((benign_test, insitu_test, invasive1_test, invasive2_test, invasive3_test), axis = 0)
len_train = len(X_train)
len_test = len(X_test)
X = np.concatenate((X_train, X_test), axis=0)
############### Labels ##############
s=100
Y_test = []
Y_train = []
for folder in  os.listdir(path +'/TrainingSet') : 
    files = gb.glob(pathname= str( path +'/TrainingSet//' + folder + '/*.png'))
    for file in files: 
        Y_train.append(code[folder])

for folder in  os.listdir(path +'/ValidationSet') : 
    files = gb.glob(pathname= str( path +'/ValidationSet//' + folder + '/*.png'))
    for file in files: 
        Y_test.append(code[folder])

Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
Y = np.concatenate((Y_train, Y_test), axis=0)
############################ Preprocessing ############################
# run create_feature_matrix on our dataframe of images
X = create_feature_matrix(X)

################ Scale feature matrix & PCA ################
# get shape of feature matrix
print('Feature matrix shape is: ', X.shape)

# define standard scaler
ss = StandardScaler()
# run this on our feature matrix
X_scaled = ss.fit_transform(X)
pca = PCA(n_components=500)
# use fit_transform to run PCA on our standardized matrix
X_pca = ss.fit_transform(X_scaled)

# look at new shape
print('PCA matrix shape is: ', X_pca.shape)

################### Train Test Split ###################
X_valid= X[len_train:len_train+len_test,]
Y_valid = Y_test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=1234123)


################### Training the Model ###################
# define support vector classifier
svm = SVC(kernel='linear', probability=True, random_state=42)
# fit model
svm.fit(X_train, Y_train)

#Save SVM object
pickle_dump(svm, path='./checkpoints/SVM_Model')


################### Evaluate Model ###################   
# generate predictions
y_pred = svm.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(Y_test, y_pred)
print('Model accuracy is: ', accuracy)     


###################  ROC curve & AUC ###################       
# predict probabilities for X_test using predict_proba
probabilities = svm.predict_proba(X_test)

# select the probabilities for label 1.0
y_proba = probabilities[:, 1]

# calculate false positive rate and true positive rate at different thresholds
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_proba, pos_label=1)

# calculate AUC
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
# plot the false positive rate on the x axis and the true positive rate on the y axis
roc_plot = plt.plot(false_positive_rate,
                    true_positive_rate,
                    label='AUC = {:0.2f}'.format(roc_auc))

plt.legend(loc=0)
plt.plot([0,1], [0,1], ls='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate');

################### Confusion Matrix ###################

print('Shape of X_valid:'+str(np.shape(X_test)))
print('Shape of Y_valid:'+str(np.shape(Y_test)))

# generate predictions
y_pred = svm.predict(X_test)

cm = confusion_matrix(Y_test, y_pred)
# create confusion matrix
plot_confusion_matrix(cm, list(code.keys()), 'confusion_matrix_SVM.png')
