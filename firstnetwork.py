# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:33:42 2019

@author: FridaL
"""

# Import data
#import h5py
#with h5py.File('test_X.mat', 'r') as file:
#    test_X = list(file['test_X'])
#    
#import numpy as np    
#test_X = np.array(test_X)  
#test_X = np.transpose(test_X)
#test_X = test_X.astype('float32')
#
#import h5py
#with h5py.File('test_Y.mat', 'r') as file:
#    test_Y = list(file['test_Y'])
#    
#   
#test_Y = np.array(test_Y)  
#test_Y = np.transpose(test_Y)
#test_Y = test_Y.astype('float32')
#
#import h5py
#with h5py.File('train_X.mat', 'r') as file:
#    train_X = list(file['train_X'])
#    
  
#train_X = np.array(train_X)  
#train_X = np.transpose(train_X)
#train_X = train_X.astype('float32')
#
#import h5py
#with h5py.File('train_Y.mat', 'r') as file:
#    train_Y = list(file['train_Y'])
#    
#
#train_Y = np.array(train_Y)  
#train_Y = np.transpose(train_Y)
#train_Y = train_Y.astype('float32')

#from keras.utils import to_categorical
#
#train_Y_one_hot = to_categorical(train_Y)
#test_Y_one_hot = to_categorical(test_Y)
#
## Find unique number of training examples as classes
#classes = np.unique(train_Y)
#nClasses = len(classes)
#
## Reshaping it
#train_X = train_X.reshape(6000,76800, 1)
#test_X = test_X.reshape(955, 76800, 1)
#
## Split into training and validation data (80 train, 20 test)
#from sklearn.model_selection import train_test_split
#train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)


""" Build CNN & LSTM """

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 1
num_classes = 10

model = Sequential()
model.add(Conv1D(100,10,activation='relu',input_shape=(76800,1)))
model.add(MaxPooling1D(3))
model.add(LSTM(10))
model.add(Dense(11, activation='softmax'))

print(model.summary())

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
#
## Train the model 
model = model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
#
## Evaluation of model 
#test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=0)
#
#print('Test loss:', test_eval[0])
#print('Test accuracy:', test_eval[1])
#
