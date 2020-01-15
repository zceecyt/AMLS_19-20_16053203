from A1.gender import landmarks, landmarks_test
from A2.gender import landmarks_v2, landmarks_v2_test
import B1.gender
import B2.gender
import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
import pandas as pd
import time
start_time = time.time()
import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy import stats
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, ShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import preprocessing

# ======================================================================================================================
# Data preprocessing
data_train, data_val, data_test = data_preprocessing(args...)
# ======================================================================================================================
# Task A1
model_A1 = A1(args...)                 # Build model object.
acc_A1_train = model_A1.train(args...) # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A1_test = model_A1.test(args...)   # Test model based on the test set.
Clean up memory/GPU etc...             # Some code to free memory if necessary.


# ======================================================================================================================
# Task A2
model_A2 = A2(args...)
acc_A2_train = model_A2.train(args...)
acc_A2_test = model_A2.test(args...)
Clean up memory/GPU etc...


# ======================================================================================================================
# Task B1
model_B1 = B1(args...)
acc_B1_train = model_B1.train(args...)
acc_B1_test = model_B1.test(args...)
Clean up memory/GPU etc...


# ======================================================================================================================
# Task B2
model_B2 = B2(args...)
acc_B2_train = model_B2.train(args...)
acc_B2_test = model_B2.test(args...)
Clean up memory/GPU etc...


# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'
# acc_A1_test = 'TBD'