### General training file ###

### JA: in future: convert into trainer class: like in lecture ML iteration CH4

### Imports ###

# General libraries
import numpy as np
import pandas as pd

# Image related
import matplotlib.pyplot as plt
from matplotlib import image
from keras.preprocessing.image import load_img, img_to_array
import os

# CNN
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import Sequential, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


def preprocess(image_df):
    """function that pre-process the data"""
    # Split Training Set
    train_set = image_df[(image_df.Usage == 'Training')]
    val_set = image_df[(image_df.Usage == 'PublicTest')]
    test_set = image_df[(image_df.Usage == 'PrivateTest')]
    X_train = np.array(list(map(str.split, train_set.pixels)), np.float32)
    X_val = np.array(list(map(str.split, val_set.pixels)), np.float32)
    X_test = np.array(list(map(str.split, test_set.pixels)), np.float32)
    # Reshape X
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
    X_val = X_val.reshape(X_val.shape[0], 48, 48, 1)
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
    # Define y
    y_train = train_set["emotion"]
    y_val = val_set["emotion"]
    y_test = test_set["emotion"]
    # One Hot Encode our Target for TensorFlow processing
    y_cat_train = to_categorical(y_train, num_classes=7)
    y_cat_test = to_categorical(y_test, num_classes=7)
    y_cat_val = to_categorical(y_val, num_classes=7)
    return X_train, y_cat_train

def initialize_model_bl():
    """function that pre-process the data"""

    model_bl = models.Sequential()

    model_bl.add(layers.Conv2D(filters = 16, kernel_size=(3, 3), strides=(2,2), activation='relu', padding='same', input_shape=(48, 48, 1)))
    model_bl.add(layers.MaxPool2D(pool_size=(3,3)))

    model_bl.add(layers.Conv2D(32, kernel_size=(2,2), activation='relu'))
    model_bl.add(layers.MaxPool2D(pool_size=(2,2)))

    model_bl.add(layers.Conv2D(32, kernel_size=(2,2), activation='relu'))
    model_bl.add(layers.MaxPool2D(pool_size=(2,2)))

    model_bl.add(layers.Flatten())
    model_bl.add(layers.Dense(7, activation='softmax'))

    return model_bl


def compile_model_bl(model_bl):
    """function that compiles the model"""
    model_bl.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model_bl

def fit_model_bl(model_bl, X_train, y_cat_train):
    """function that fits the model"""
    es = EarlyStopping(patience=20)
    history_bl = model_bl.fit(X_train, y_cat_train,
                    epochs=200,
                    batch_size=32,
                    verbose=1,
                    validation_split=0.3,
                    callbacks=[es])
    return history_bl


"""
if __name__ == '__main__':
    # get training data from GCP bucket
    df = get_data()

    # preprocess data
    X_train, y_train = preprocess(df)

    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)
    reg = train_model(X_train, y_train)

    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model(reg)

"""
