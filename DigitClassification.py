# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 21:48:37 2020

@author: rajes
"""
import cv2,os

import numpy as np
import csv
import glob
from scipy import io

from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt  
from sklearn import svm
from sklearn import metrics
import joblib
from sklearn.decomposition import PCA
import numpy as np
from sklearn.utils import shuffle

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten



#dirList=[]
X=[]
Y=[]
X1=[]
for claasss in ['1'] :
 #un this script for all labels
    dirList=(glob.glob("C:/Users/rajes/Masters Degree Studies/Pattern Learning &Machine Learning/Assignment 1/tau-ethiopic-digit-recognition/train/train/"+claasss+"/*.jpg"))
     
    for img_path in dirList : 
         
        im = cv2.imread(img_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im= cv2.resize(im, (28, 28), interpolation=cv2.INTER_AREA)
        X1.append(np.array(im))
        #Y.append(int(claasss))

X1=np.array(X1)
Y=np.array(Y)

datadict = {
    'X':X,
    'Y':Y
    }
io.savemat('train.mat',datadict)

X_test=[]
img=[]
dirList=(glob.glob("C:/Users/rajes/Masters Degree Studies/Pattern Learning &Machine Learning/Assignment 1/tau-ethiopic-digit-recognition/test/test/*.jpg"))
 
for img_path in dirList : 
    
    img.append( int(img_path.split("/")[8].split(".")[0].split("\\")[1]))
   
    im = cv2.imread(img_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im= cv2.resize(im, (28, 28), interpolation=cv2.INTER_AREA)
    X_test.append(np.array(im))

X_test=np.array(X_test)

datadict = {
    'X':X_test,
    'Y':Y
    }
io.savemat('test.mat',datadict)



#X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


#x_test = mat_test['X']


num_pixels = X.shape[1] * X.shape[2]

X = X.reshape((X.shape[0], 28,28,1)).astype('float32')


X_test = X_test.reshape((X_test.shape[0], 28,28,1)).astype('float32')

# normalize inputs from 0-255 to 0-1
X = X/ 255
X_test = X_test / 255
# one hot encode outputs

Y = np_utils.to_categorical(Y)
#Y_test = np_utils.to_categorical(Y_test)

num_classes = Y.shape[1]

# # # define baseline model
# # model = Sequential()
# # model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
# # model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X, Y ,  epochs=50, batch_size=200, verbose=2)

#scores = model.evaluate(X_test, Y_test, verbose=0)
X1 = X1.reshape((X1.shape[0], 28,28,1)).astype('float32')
X1 = X1/255
Y_pred=model.predict_classes(X_test)

for i in Y_pred :
    print(i)


for i in range(0,9999) :
    output=[]
    output.append(img[i])
    output.append(Y_pred[i])
    
    with open('predictions.csv', 'a', newline = '') as f1 :
     writer = csv.writer(f1)
     writer.writerow(output)

#joblib.dump(model, "model/svm_0to5label_linear_2") 

#print("Baseline Error: %.2f%%" % (100-scores[1]*100))

print ("Getting Accuracy .....")


# model = svm.SVC(gamma=0.001)

# print ("Fitting this might take some time .....")

# model.fit(X_train,Y_train)


# #joblib.dump(model, "model/svm_0to5label_linear_2") 

# print ("predicting .....")
# predictions = model.predict(X_test)

# print ("Getting Accuracy .....")

# print(metrics.accuracy_score(Y_test, predictions))