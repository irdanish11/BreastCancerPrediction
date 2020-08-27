# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 13:46:24 2020

@author: junaid
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import glob as gb
import cv2
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
from keras.applications import DenseNet201
from PIL import Image
import seaborn
from sklearn.metrics import confusion_matrix


##Build Model
def build_model(backbone, classes, lr = 1e-4):
  classifier = Sequential()
  classifier.add(backbone)
  classifier.add(layers.GlobalAveragePooling2D())
  classifier.add(layers.Dropout(0.5))
  classifier.add(layers.BatchNormalization())
  classifier.add(layers.Dense(classes, activation = 'softmax'))
  
  classifier.compile(loss = 'sparse_categorical_crossentropy', 
                     optimizer=Adam(lr=lr),
                     metrics=['accuracy'])
  
  return classifier


def get_model(ckpt_path, classes):
    resnet = DenseNet201(weights='imagenet',
                         include_top=False,
                         input_shape=(100,100,3),
                         classes = classes)
    model = build_model(resnet, classes, lr = 1e-4)
    model.summary()
    ##Load Trained Weights
    model.load_weights(ckpt_path)
    return model


def get_pred(path, model, return_img=False):
    image = np.asarray(Image.open(path).convert("RGB"))
    test_image = cv2.resize(image, (100, 100))
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    if return_img:
        return result, image
    return result

def predict_class(path, code, model, draw=True):
    pred, image = get_pred(path, model, return_img=True)
    pred_class = code[np.argmax(pred)]
    if draw:
        plt.title(label = 'Orignal image name: {0} \n'.format(path.split('/')[-1]) + '\n Predicted Class: ' + pred_class)
        plt.imshow(image)
    return pred_class
    
def read_imgs(files):
    class_test = []
    for file in files:
        image = np.asarray(Image.open(file).convert("RGB"))
        test_image = cv2.resize(image, (100, 100))
        class_test.append(test_image)
    class_test = np.array(class_test)
    return class_test
 
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
       
def bar_plot(acc, output_filename):        
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    labels = list(acc.keys())
    val = list(acc.values())
    ax.barh(labels, val)
    ax.set_xlabel('Accuracy of Class')
    ax.set_title('Class Wise Accuracy')
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
def get_accuracy(path, code, model, bar_chart=True, confusion_mat=True, verbose=True):
    inv_code = {v: k for k, v in code.items()}
    acc = {}
    i=0
    for folder in  os.listdir(path) : 
        files = gb.glob(pathname= path+'/'+folder+'/*.png')
        class_test = read_imgs(files)
        class_label = np.full(len(files), inv_code[folder])
        acc[folder] = model.evaluate(class_test, class_label)[1]
        if i==0:
            X_test = class_test
            y_test = class_label
        else:
            X_test = np.append(X_test, class_test, axis=0)
            y_test = np.append(y_test, class_label, axis=0)
        i += 1
    acc['AllData'] = model.evaluate(X_test, y_test)[1]
    if bar_chart:
        bar_plot(acc, 'bar_chart.png')
    if confusion_mat:
        cf = confusion_matrix(y_test, model.predict_classes(X_test))
        # create confusion matrix
        plot_confusion_matrix(cf, list(acc.keys()), 'confusion_matrix.png')
    if verbose:
        print('\n\n\t\t________________ Class Wise Accuracy ________________')
        for k in list(acc.keys()):
            print('Accuracy on {0}: {1}'.format(k, acc[k]))
    return acc        

################ Loading Model ################       
model = get_model(ckpt_path='checkpoints/5Class_Model_Data2.hdf5', classes=5)
##Define Code for Predictions
code = {0 :'Benign', 1 :'Insitu', 2 :'InvGrade1', 3 :'InvGrade2', 4 :'InvGrade3'}
code_M = {0 :'Benign', 1 :'Insitu', 2 :'Invasive'}


################### Test Single Image #####################
img_path = 'Data/ValidationSet2/InvGrade2/SOB_M_DC-14-12312-40-032.png'
pred_class = predict_class(img_path, code, model, draw=True)

######### Get Accuracy and Confusion Matrix Plots #########
#path of validation data
path = './Data/ValidationSet2'
acc = get_accuracy(path, code, model, bar_chart=True, confusion_mat=True, verbose=True)


