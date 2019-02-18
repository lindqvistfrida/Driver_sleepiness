# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 11:07:10 2019

@author: FridaL
"""

""" TEST """
import h5py as h5py_
import numpy as np 

file = h5py_.File('test_X_sub_30_EEG.mat', 'r')
test_X_EEG = file['test_X']
       
test_X_EEG = np.array(test_X_EEG)  
test_X_EEG = np.transpose(test_X_EEG)
test_X_EEG = test_X_EEG.astype('float32')


file = h5py_.File('test_X_sub_30_ECG.mat', 'r')
test_X_ECG = file['test_X']
      
test_X_ECG = np.array(test_X_ECG)  
test_X_ECG = np.transpose(test_X_ECG)
test_X_ECG = test_X_ECG.astype('float32')

file = h5py_.File('test_X_sub_30_EOG.mat', 'r')
test_X_EOG = file['test_X']
    
test_X_EOG = np.array(test_X_EOG)  
test_X_EOG = np.transpose(test_X_EOG)
test_X_EOG = test_X_EOG.astype('float32')


file = h5py_.File('test_Y_sub_30_5classes.mat', 'r')
test_Y = file['test_Y']
    
test_Y = np.array(test_Y)  
test_Y = np.transpose(test_Y)
test_Y = test_Y.astype('float32')

""" TRAIN """

file = h5py_.File('train_X_sub_30_EEG.mat', 'r')
train_X_EEG = file['train_X']
        
train_X_EEG = np.array(train_X_EEG)  
train_X_EEG = np.transpose(train_X_EEG)
train_X_EEG = train_X_EEG.astype('float32')


file = h5py_.File('train_X_sub_30_ECG.mat', 'r')
train_X_ECG = file['train_X']
      
train_X_ECG = np.array(train_X_ECG)  
train_X_ECG = np.transpose(train_X_ECG)
train_X_ECG = train_X_ECG.astype('float32')


file = h5py_.File('train_X_sub_30_EOG.mat', 'r')
train_X_EOG = file['train_X']
       
train_X_EOG = np.array(train_X_EOG)  
train_X_EOG = np.transpose(train_X_EOG)
train_X_EOG = train_X_EOG.astype('float32')

file = h5py_.File('train_Y_sub_30_5classes.mat', 'r')
train_Y = file['train_Y']
    
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
train_X_ECG = train_X_ECG.reshape(66980,7680, 1)
train_X_EEG = train_X_EEG.reshape(66980,7680, 1)
train_X_EOG = train_X_EOG.reshape(66980,7680, 1)
test_X_ECG = test_X_ECG.reshape(13360, 7680, 1)
test_X_EEG = test_X_EEG.reshape(13360, 7680, 1)
test_X_EOG = test_X_EOG.reshape(13360,7680,1)

# Split into training and validation data (80 train, 20 test)
#from sklearn.model_selection import train_test_split
#[train_X_ECG,train_X_EEG,train_X_EOG],[valid_X_ECG,valid_X_EEG,valid_X_EOG],train_label,valid_label = train_test_split([train_X_ECG,train_X_EEG,train_X_EOG], train_Y_one_hot, test_size=0.2, random_state=13)
#train_X_EEG,valid_X_EEG,train_label,valid_label_EEG = train_test_split(train_X_EEG, train_Y_one_hot, test_size=0.2, random_state=13)
#train_X_EOG,valid_X_EOG,train_label,valid_label_EOG = train_test_split(train_X_EOG, train_Y_one_hot, test_size=0.2, random_state=13)

import numpy as np
import keras
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Concatenate, Conv1D, MaxPooling1D, LSTM, BatchNormalization

batch_size = 64
epochs = 1

inp1 = Input(shape=train_X_ECG.shape[1:])
inp2 = Input(shape=train_X_EEG.shape[1:])
inp3 = Input(shape=train_X_EOG.shape[1:])

conv1 = Conv1D(64, 3, activation='relu',input_shape=(7680,1))(inp1)
conv2 = Conv1D(64, 3, activation='relu',input_shape=(7680,1))(inp2)
conv3 = Conv1D(64, 3, activation='relu',input_shape=(7680,1))(inp3)

BN1 = BatchNormalization()(conv1)
BN2 = BatchNormalization()(conv2)
BN3 = BatchNormalization()(conv3)

conv11 = Conv1D(64, 3, activation='relu')(BN1)
conv12 = Conv1D(64, 3, activation='relu')(BN2)
conv13 = Conv1D(64, 3, activation='relu')(BN3)

maxp11 = MaxPooling1D(2)(conv11)
maxp12 = MaxPooling1D(2)(conv12)
maxp13 = MaxPooling1D(2)(conv13)

DO1 = Dropout(0.25)(maxp11)
DO2 = Dropout(0.25)(maxp11)
DO3 = Dropout(0.25)(maxp11)

conv21 = Conv1D(64, 3, activation='relu')(DO1)
conv22 = Conv1D(64, 3, activation='relu')(DO2)
conv23 = Conv1D(64, 3, activation='relu')(DO3)

maxp21 = MaxPooling1D(2)(conv21)
maxp22 = MaxPooling1D(2)(conv22)
maxp23 = MaxPooling1D(2)(conv23)

#flt1 = Flatten()(maxp1)
#flt2 = Flatten()(maxp2)
#flt3 = Flatten()(maxp3)

mrg = Concatenate(axis=-1)([maxp21,maxp22,maxp23])
lstm = LSTM(10)(mrg)
#dense = Dense(256, activation='relu')(mrg)
op = Dense(6, activation='softmax')(lstm)
model = Model(inputs=[inp1, inp2, inp3], outputs=op)

print(model.summary())

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
# Train the model 
model.fit([train_X_ECG,train_X_EEG,train_X_EOG], train_Y_one_hot, batch_size=batch_size,epochs=epochs,verbose=1,validation_split=0.2)

# Evaluation of model 
test_eval = model.evaluate([test_X_ECG, test_X_EEG, test_X_EOG], test_Y_one_hot, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


