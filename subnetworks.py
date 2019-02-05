# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 11:07:10 2019

@author: FridaL
"""

import h5py
with h5py.File('test_X_EEG.mat', 'r') as file:
    test_X_EEG = list(file['test_X_EEG'])
    
import numpy as np    
test_X_EEG = np.array(test_X_EEG)  
test_X_EEG = np.transpose(test_X_EEG)
test_X_EEG = test_X_EEG.astype('float32')

import h5py
with h5py.File('test_X_ECG.mat', 'r') as file:
    test_X_ECG = list(file['test_X_ECG'])
    
import numpy as np    
test_X_ECG = np.array(test_X_ECG)  
test_X_ECG = np.transpose(test_X_ECG)
test_X_ECG = test_X_ECG.astype('float32')

import h5py
with h5py.File('test_Y.mat', 'r') as file:
    test_Y = list(file['test_Y'])
    
test_Y = np.array(test_Y)  
test_Y = np.transpose(test_Y)
test_Y = test_Y.astype('float32')

import h5py
with h5py.File('train_X_EEG.mat', 'r') as file:
    train_X_EEG = list(file['train_X_EEG'])
    
import numpy as np    
train_X_EEG = np.array(train_X_EEG)  
train_X_EEG = np.transpose(train_X_EEG)
train_X_EEG = train_X_EEG.astype('float32')

import h5py
with h5py.File('train_X_ECG.mat', 'r') as file:
    train_X_ECG = list(file['train_X_ECG'])
    
import numpy as np    
train_X_ECG = np.array(train_X_ECG)  
train_X_ECG = np.transpose(train_X_ECG)
train_X_ECG = train_X_ECG.astype('float32')

import h5py
with h5py.File('train_Y.mat', 'r') as file:
    train_Y = list(file['train_Y'])
    
train_Y = np.array(train_Y)  
train_Y = np.transpose(train_Y)
train_Y = train_Y.astype('float32')

from keras.utils import to_categorical

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Find unique number of training examples as classes
classes = np.unique(train_Y)
nClasses = len(classes)

# Reshaping it
train_X_EEG = train_X_EEG.reshape(2000,7680, 1)
train_X_ECG = train_X_ECG.reshape(2000,7680, 1)
test_X_EEG = test_X_EEG.reshape(500, 7680, 1)
test_X_ECG = test_X_ECG.reshape(500, 7680, 1)

# Split into training and validation data (80 train, 20 test)
#from sklearn.model_selection import train_test_split
#train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

import numpy as np
import keras
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Concatenate, Conv1D, MaxPooling1D

inp1 = Input(shape=train_X_EEG.shape[1:])
inp2 = Input(shape=train_X_ECG.shape[1:])

conv1 = Conv1D(64, 3, activation='relu',input_shape=(7680,1))(inp1)
conv2 = Conv1D(64, 3, activation='relu',input_shape=(7680,1))(inp2)

maxp1 = MaxPooling1D(3)(conv1)
maxp2 = MaxPooling1D(3)(conv2)

flt1 = Flatten()(maxp1)
flt2 = Flatten()(maxp2)

mrg = Concatenate(axis=-1)([flt1,flt2])
dense = Dense(256, activation='relu')(mrg)
op = Dense(10, activation='softmax')(dense)
model = Model(inputs=[inp1, inp2], outputs=op)

print(model.summary())

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

model.fit([train_X_EEG,train_X_ECG], train_Y_one_hot, nb_epoch=1, batch_size=28)

# Evaluation of model 
test_eval = model.evaluate([test_X_EEG, test_X_ECG], test_Y_one_hot, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


