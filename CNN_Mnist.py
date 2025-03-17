# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 12:01:00 2021

Python code in which a CNN is designed and trained to recognise MNIST dataset elements
The trained model is created to be implemented on STM32 board (STM32F411)

@author: RJ264980
"""

import sys, os, array, time
import numpy as np
import matplotlib.pyplot as plt
import IPython

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


class timer:
    def __init__(self, name=None):
        self.name = name
        self.T_start = -1
        self.T_stop  = -1

    def tic(self):
        self.T_start = time.time()

    def toc(self):
        self.T_stop = time.time()

    def res(self):
        if (self.T_start == -1) or (self.T_stop == -1):
            print("Error: Measurement cannot be done")
        else:
            return str(self.T_stop - self.T_start)



def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def load_mnist_data(path):
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

def build_model(data):

        # Small CNN for MNIST recognition
        model = models.Sequential()
        
        # Dense layer
        model.add(layers.Conv2D(2, (3, 3), padding='same', activation='relu', input_shape=data.input_shape))
        model.add(layers.MaxPooling2D((2, 2), padding='valid'))
        model.add(layers.Flatten())
        
        # Dense layer
        model.add(layers.Dense(16, activation='relu'))
        
        # Output layer
        model.add(layers.Dense(10, activation='softmax'))
                
        return model
    
class dataset:
    def __init__(self):
        with np.load('mnist.npz') as f:
            self.x_train, self.y_train = f['x_train'], f['y_train']
            self.x_test, self.y_test = f['x_test'], f['y_test']
        #(self.x_train, self.y_train), (self.x_test, self.y_test)  = load_data('mnist.npz')
        
        # Rescale of images
        self.x_train = self.x_train / 255.0
        self.x_test  = self.x_test / 255.0
        
        # Take 10.000 image from x_train to constitue a validation dataset
        self.x_val = self.x_train[50000:]
        self.y_val = self.y_train[50000:]
        
        self.x_train = self.x_train[:50000]
        self.y_train = self.y_train[:50000]
        
        # Reshape of x_train et x_test
        #if tf.keras.datasets.mnist.image_data_format == 'channel_first':
        #self.x_train = self.x_train.reshape((self.x_train.shape[0], 1, self.x_train.shape[1], self.x_train.shape[2]))
        #self.x_val   = self.x_val.reshape((self.x_val.shape[0], 1, self.x_val.shape[1], self.x_val.shape[2]))
        #self.x_test  = self.x_test.reshape((self.x_test.shape[0], 1, self.x_test.shape[1], self.x_test.shape[2]))
        #self.input_shape = (1, 28, 28)
        
        #else:
        self.x_train = self.x_train.reshape((self.x_train.shape[0], self.x_train.shape[1], self.x_train.shape[2], 1))
        self.x_val   = self.x_val.reshape((self.x_val.shape[0], self.x_val.shape[1], self.x_val.shape[2], 1))
        self.x_test  = self.x_test.reshape((self.x_test.shape[0], self.x_test.shape[1], self.x_test.shape[2], 1))
        self.input_shape = self.x_train.shape[1:]
        
        
        # Transform label to one hot vector
        self.y_train = tf.keras.utils.to_categorical(self.y_train, 10)
        self.y_val   = tf.keras.utils.to_categorical(self.y_val, 10)
        self.y_test  = tf.keras.utils.to_categorical(self.y_test, 10)
        
        self.nb_epochs  = 5
        self.batch_size = 128
        
        print("Number training examples:  ", len(self.x_train))
        print("Number test examples:      ", len(self.x_test))
        print("Number validation examples:", len(self.x_val))
        print("\n")
        print("\tTrain Dataset      --> x_train: " + str(np.shape(self.x_train)) + "    y_train: " + str(np.shape(self.y_train)))
        print("\tValidation Dataset --> x_val:   " + str(np.shape(self.x_val))   + "    y_val:   " + str(np.shape(self.y_val)))
        print("\tTesting Dataset    --> x_test:  " + str(np.shape(self.x_test))  + "    y_test:  " + str(np.shape(self.y_test)))
        print("\tNumber of epochs:  "+str(self.nb_epochs))
        print("\tBatch size:        "+str(self.batch_size))
        print("\n")
        
    def MLP_input_data_preparation(self):
        self.input_shape = np.prod(self.x_train.shape[1:])
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.input_shape)
        self.x_val   = self.x_val.reshape(self.x_val.shape[0], self.input_shape)
        self.x_test  = self.x_test.reshape(self.x_test.shape[0], self.input_shape)
        
        print ("\n")
        print ("New dimensions after MLP reshape:\n")
        print ("Train Dataset      --> x_train: " + str(np.shape(self.x_train)) + "    y_train: " + str(np.shape(self.y_train)))
        print ("Validation Dataset --> x_val:   " + str(np.shape(self.x_val))   + "    y_val:   " + str(np.shape(self.y_val)))
        print ("Testing Dataset    --> x_test:  " + str(np.shape(self.x_test))  + "    y_test:  " + str(np.shape(self.y_test)))
        print ("\n")
    
        

def train_model(dataset):
    
    print("Preparing context to train of the model ...")
    chronos = timer()

    x_train     = dataset.x_train
    x_val       = dataset.x_val
    y_train     = dataset.y_train
    y_val       = dataset.y_val
    batch_size  = dataset.batch_size
    epochs      = dataset.nb_epochs
    
    l_rate = 0.01
    optimizer = tf.keras.optimizers.Adam(lr=l_rate)
    
    model = build_model(dataset)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    
    chronos.tic()
    print("### START TRAINING ###")
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        shuffle=True)
    
    chronos.toc()
    print("\n### STOP TRAINING ###")

    train_scores = model.evaluate(x_train, y_train, verbose=0)
    val_scores = model.evaluate(x_val, y_val, verbose=0)

    print("\n")
    print('***** Train loss:    ', train_scores[0])
    print('***** Train accuracy:', train_scores[1])
    print("")
    print('***** Val loss:    ', val_scores[0])
    print('***** Val accuracy:', val_scores[1])
    print("\n")
    
    return model



def test_model(dataset, trained_model, save_pred=False, trust_indicator=False):
    chronos = timer()
    
    x_test = dataset.x_test
    y_test = dataset.y_test
    
    print("### START TESTING ###")
    chronos.tic()
        
    test_score = trained_model.evaluate(x_test, y_test, verbose=0)
    
    chronos.toc()
    print("")
    print('***** Test loss:    ', test_score[0])
    print('***** Test accuracy:', test_score[1])
    print("")
    
    if trust_indicator:
        output_pred = trained_model.predict(x_test)
        output_pred.sort(axis=1)
        
        trust_diff = [(output_pred[i][-1] - output_pred[i][-2]) for i in range(x_test.shape[0])]
        trust_diff = np.array(trust_diff)
        
        trust_mean = trust_diff.mean(axis=0)
        trust_min  = np.amin(trust_diff)
        
        print("")
        print("***** Trust difference mean:", trust_mean)
        print("***** Minimal difference   :", trust_min)
        print("")


        
if __name__ == '__main__':
    
    save_files = True
    print("Start process ... \n")
    
    data = dataset()
    
    # Training of Neural Network
    trained_model = train_model(data)
    test_model(data, trained_model)
    
    # Saving model and testing datasets
    if save_files == True:
        np.save("MNIST_xtest_NN_C2_16_10.npy", data.x_test)
        np.save("MNIST_ytest_NN_C2_16_10.npy", data.y_test)
        trained_model.save("MNIST_NN_C2_16_10.h5")
    
    
    
    