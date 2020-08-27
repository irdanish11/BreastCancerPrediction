# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 12:27:20 2020

@author: junaid
"""
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras import layers
from keras.optimizers import Adam
from keras.applications import DenseNet201
import numpy as np
import os
import glob as gb
import cv2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tqdm import tqdm
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

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

path = './Data/Invaisve'
Invasive1_train = np.array(Dataset_loader(path+'/TrainingSet/InvGrade1',100))
Invasive2_train = np.array(Dataset_loader(path+'/TrainingSet/InvGrade2',100))
Invasive3_train = np.array(Dataset_loader(path+'/TrainingSet/InvGrade3',100))

Invasive1_test = np.array(Dataset_loader(path+'/ValidationSet/InvGrade1',100))
Invasive2_test = np.array(Dataset_loader(path+'/ValidationSet/InvGrade2',100))
Invasive3_test = np.array(Dataset_loader(path+'/ValidationSet/InvGrade3',100))

##Creating Lables and Preparing Training and Testing Datasets

# Creating X_test and X_train 
X_train = np.concatenate((Invasive1_train, Invasive2_train, Invasive3_train), axis = 0)

X_test = np.concatenate((Invasive1_test, Invasive2_test, Invasive3_test), axis = 0)



# Creating Labels for Y_test and Y_trian
code = {'InvGrade1':0 ,'InvGrade2':1 ,'InvGrade3':2}

def getcode(n) : 
    for x , y in code.items() : 
        if n == y : 
            return x

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

#Spliting data for Training and Evalutation
# x_train, x_val, y_train, y_val = train_test_split(
#     X_train, Y_train, 
#     test_size=0.1, 
#     random_state=11)

#Data Generator

BATCH_SIZE = 16
# Using original generator
train_generator = ImageDataGenerator(zoom_range=2,  # set range for random zoom
                                     rotation_range = 90,
                                     horizontal_flip=True,  # randomly flip images
                                     vertical_flip=True,  # randomly flip images
                                     )
#--------------------Building Model------------------------

def build_model(backbone, lr = 1e-4):
  classifier = Sequential()
  classifier.add(backbone)
  classifier.add(layers.GlobalAveragePooling2D())
  classifier.add(layers.Dropout(0.5))
  classifier.add(layers.BatchNormalization())
  classifier.add(layers.Dense(3, activation = 'softmax'))
  
  classifier.compile(loss = 'sparse_categorical_crossentropy', 
                     optimizer=Adam(lr=lr),
                     metrics=['accuracy'])
  
  return classifier

resnet = DenseNet201(weights='imagenet',
                     include_top=False,
                     input_shape=(100,100,3),
                     classes = 3)

model = build_model(resnet, lr = 1e-4)
model.summary()

learn_control = ReduceLROnPlateau(monitor = 'val_loss', 
                                  patience = 5, verbose = 1, 
                                  factor = 0.2, min_lr = 1e-7)


##Training the Model
ckpt_path = './checkpoints/Irfan'
ckpt_name = "Invasive_Model.hdf5"
os.makedirs(ckpt_path, exist_ok = True)
ckpt = ModelCheckpoint(ckpt_path + '/' + ckpt_name, monitor = 'val_loss', save_best_only = True) 
es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit_generator(train_generator.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                              steps_per_epoch = X_train.shape[0]/BATCH_SIZE,
                              epochs = 50,
                              validation_data=(X_test, Y_test),
                              callbacks = [learn_control, ckpt, es])

#///////////////////Performance Matrics///////////////////#


#///////////////////Prediction///////////////////#


#--------------Confusion Matrix----------#
