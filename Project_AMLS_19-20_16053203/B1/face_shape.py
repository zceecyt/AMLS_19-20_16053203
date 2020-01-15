import time
start_time = time.time()
import os
import numpy as np
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import adam
from keras import models
import cv2
import dlib
import pandas as pd
import matplotlib.pyplot as plt
import sys
from tqdm.notebook import tqdm_notebook
from sklearn import svm, datasets
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, ShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

def preprocess(train_path, test_path):
    df = pd.read_csv(train_path + 'labels.csv', sep = '\t')
    df = df.drop(columns = [df.columns[0]]).drop(columns = [df.columns[1]])
    df['face_shape'] = df['face_shape'].apply(str)

    df2 = pd.read_csv(test_path + 'labels.csv', sep = '\t')
    df2 = df2.drop(columns = [df2.columns[0]]).drop(columns = [df2.columns[1]])
    df2['face_shape'] = df2['face_shape'].apply(str)
    
    training, testing = np.split(df.sample(frac=1), [int(0.9*len(df)),]) #splitting at n-array

    img = (train_path + 'img')
    img2 = (test_path + 'img')

    # set up data generator
    data_generator = ImageDataGenerator(
        rescale = 1./255.,
        validation_split = 0.2,
        horizontal_flip=True,
        vertical_flip=True   
    )

    # Get batches of training dataset from the dataframe
    print("Training Dataset Preparation: ")
    train_generator = data_generator.flow_from_dataframe(
            dataframe = training, directory = img,
            x_col = "file_name", y_col = "face_shape",
            class_mode = 'categorical', target_size = (64,64),
            batch_size = 128, subset = 'training') 

    # Get batches of validation dataset from the dataframe
    print("\nValidation Dataset Preparation: ")
    validation_generator = data_generator.flow_from_dataframe(
            dataframe = training, directory = img ,
            x_col = "file_name", y_col = "face_shape",
            class_mode = 'categorical', target_size = (64,64),
            batch_size = 128, subset = 'validation')
    
    return train_generator, validation_generator, data_generator, df2, img2

def CNN_modelling():
    my_model= models.Sequential()

    # Add first convolutional block
    my_model.add(Conv2D(16, (3, 3), activation='relu', padding='same',input_shape=(64,64,3))) 
    my_model.add(MaxPooling2D((2, 2), padding='same'))
    # second block
    my_model.add(Conv2D(32, (3, 3), activation='relu', padding='same')) 
    my_model.add(MaxPooling2D((2, 2), padding='same'))
    # third block
    my_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    my_model.add(MaxPooling2D((2, 2), padding='same'))
    # fourth block
    my_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    my_model.add(MaxPooling2D((2, 2), padding='same'))

    # make predictions
    my_model.add(Flatten())
    my_model.add(Dense(5, activation='softmax'))
    # Show a summary of the model. Displaying the number of trainable parameters
    my_model.summary()
    my_model.compile(optimizer='adam', loss='categorical_crossentropy', 
                     metrics=['accuracy'])
    
    return my_model

def fitDataInModel(model_B2, train_generator, validation_generator):
    result = model_B2.fit_generator(
                                train_generator,
                                epochs=13,
                                steps_per_epoch=train_generator.samples // train_generator.batch_size,
                                validation_data=validation_generator,
                                validation_steps=validation_generator.samples // validation_generator.batch_size
                                )
    
    plt.figure(figsize=(18, 3))

    plt.subplot(131)
    plt.plot(result.history['accuracy'])
    plt.plot(result.history['val_accuracy'])
    plt.ylim([.3,1.1])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(132)
    plt.plot(result.history['loss'])
    plt.plot(result.history['val_loss'])
    plt.ylim([0,1.7])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='best')

    plt.savefig("Custom_Keras_ODSC.png", dpi=300)

import sklearn.metrics as metrics
    
def evaluate(model_B2, data_generator, df2, img2):
    test_generator = data_generator.flow_from_dataframe(
            dataframe = df2, directory = img2,
            x_col = "file_name", y_col = "face_shape",
            batch_size=1,
            class_mode='categorical', target_size=(64, 64),
            shuffle=False)
    test_steps = test_generator.samples
    print(test_steps)

    # printing training loss and accuracy
    tr_sc = model_B2.evaluate_generator(train_generator, steps = validation_generator.samples // 32, verbose=1)
    
    print('Train loss: '+ str(tr_sc[0]))
    print('Train Accuracy: '+ str(tr_sc[1]))

    test_generator.reset()
    pred = model_B2.predict_generator(test_generator, verbose=1, steps=test_steps)
    # determine the maximum activation value for each sample
    predicted_class_indices = np.argmax(pred,axis=1)
    # print(predicted_class_indices)
    # label each predicted value to correct gender
    labels = (test_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    
    cm2 = confusion_matrix(test_generator.classes, predicted_class_indices, normalize='all')

    print(cm2)
    plt.matshow(cm2)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    print('\nConfusion Matrix (normalised)')
    plt.show()
    
    test_accuracy = accuracy_score(test_generator.classes, predicted_class_indices)

    print('Classification Report\n')
    print(classification_report(test_generator.classes, predicted_class_indices))
    print('Accuracy achieved:', test_accuracy, '\u2661''\u2661''\u2661')
    
    return tr_sc[1], test_accuracy
