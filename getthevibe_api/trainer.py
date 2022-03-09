from pyexpat import model
from getthevibe_api.data import get_data_from_gcp
from termcolor import colored
import numpy as np
import pandas as pd
import joblib
from getthevibe_api.params import *
from google.cloud import storage

# CNN
from tensorflow.keras import models, layers
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
    return X_train, y_cat_train, X_val, y_cat_val

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

def fit_model_bl(model_bl, X_train, y_cat_train, X_val, y_cat_val):
    """function that fits the model"""
    es = EarlyStopping(patience=2)

    history_bl = model_bl.fit(X_train, y_cat_train,
                    epochs=1,
                    batch_size=32,
                    verbose=1,
                    validation_data=(X_val, y_cat_val),
                    callbacks=[es])
    return history_bl

def save_model(model):
    """Save the model into a .joblib and upload it on Google Storage /models folder
    HINTS : use sklearn.joblib (or jbolib) libraries and google-cloud-storage"""
    joblib.dump(model, 'model.joblib')
    print(colored("model.joblib saved locally", "green"))

def save_model_to_gcp(model, local_model_name="model.joblib"):
        """Save the model into a .joblib and upload it on Google Storage /models folder
        HINTS : use sklearn.joblib (or jbolib) libraries and google-cloud-storage"""
        # saving the trained model to disk (which does not really make sense
        # if we are running this code on GCP, because then this file cannot be accessed once the code finished its execution)
        save_model(model)
        client = storage.Client().bucket(BUCKET_NAME)
        storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
        blob = client.blob(storage_location)
        blob.upload_from_filename(local_model_name)
        print(
            "uploaded model.joblib to gcp cloud storage under \n => {}".format(
                storage_location))

if __name__ == '__main__':
    # get training data from GCP bucket
    print(colored("Get the data from the bucket", "red"))

    df = get_data_from_gcp()

    # preprocess data
    print(colored("Preprocessing...", "green"))
    X_train, y_train, X_val, y_cat_val = preprocess(df)

    # train model (locally if this file was called through the run_locally command
    # or on GCP if it was called through the gcp_submit_training, in which case
    # this package is uploaded to GCP before being executed)
    print(colored("Initializing model", "green"))
    model = initialize_model_bl()
    model = compile_model_bl(model)

    print(colored("Training the model", "red"))
    history = fit_model_bl(model,X_train, y_train, X_val, y_cat_val)

    # save trained model to GCP bucket (whether the training occured locally or on GCP)
    save_model_to_gcp(model)
