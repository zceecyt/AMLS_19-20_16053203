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
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, ShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

####################### PREPROCESSING STARTS HERE #####################

# returning feature extraction function within landmark file 
def get_data(filename):
    return filename.extract_features_labels()

def split_data(X, Y):
#   random shuffling of data 
    X, Y = shuffle(X,Y) 
#   split 70% of the dataset as training data and remaning as validation data
    tr_X, te_X, tr_Y, te_Y = train_test_split(X, Y, train_size=0.7)
    
    return tr_X, tr_Y, te_X, te_Y

def reshapeX(X):
    return X.reshape((X.shape[0], X.shape[1] * X.shape[2]))

# mapping similar index of multiple containers of data within feature labels to be used as a single entity
def reshapeY(y):
    return list(zip(*y))[0]

def preprocess(l2):    
    # returning landmark features and feature labels as X and Y respectively
    X,y = get_data(l2)
    
    # creating and self transposing array of y, offsetting y values by -1
    Y = np.array([y, -(y - 1)]).T

    tr_X, tr_Y, te_X, te_Y = split_data(X, Y)
    
    tr_X = reshapeX(tr_X)
    te_X = reshapeX(te_X)
    tr_Y = reshapeY(tr_Y)
    te_Y = reshapeY(te_Y)
    
    return tr_X, te_X, tr_Y, te_Y

####################### PARAMETER OPTIMIZATION STARTS HERE #####################

def randomSearch(X, y, param_kernel):
#   a dictionary with parameters names (string) as keys of parameters to try
    param_distributions = param_kernel 
#   number of jobs = -1 to run all processors; n_iter trades off runtime with quality of solution
#   cv is at default value for 5-fold cross validation
#   verbose gives out messages; refit is to refit an estimator to find the best parameters
#   random_state is a pseudo random number generator used for random uniform sampling from list of possible values instead of using scipy.stats distributions
    searchrand = RandomizedSearchCV(SVC(), param_distributions, n_iter=10, n_jobs=-1, refit=True, verbose=3)
    searchrand.fit(X, y)
    searchrand.cv_results_
#   returns the best parameter values of each kernel along with the kernel 
    return searchrand.best_params_, searchrand.best_estimator_ 

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
        axes.set_xlabel("Training examples")
        axes.set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes,
                           return_times=True)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
        axes.grid()
        axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes.legend(loc="best")

    return plt

def train(tr_X, te_X, tr_Y, te_Y):
    # setting upper and lower boundary values for random search range

    lin = {'C': stats.uniform(0.1, 10), 'kernel': ['linear']}
    rbf = {'C': stats.uniform(0.1, 10), 'gamma': stats.uniform(0.0001, 0.01), 'kernel': ['rbf']}
    poly = {'C': stats.uniform(0.1, 10), 'degree': stats.uniform(1, 4), 'kernel': ['poly']}

    # Obtaining optimum hyperparameters and classifier for different kernels
    linSVC_param, lin_SVC = randomSearch(tr_X, tr_Y, lin)
    rbfSVC_param, rbf_SVC = randomSearch(tr_X, tr_Y, rbf)
    polySVC_param, poly_SVC = randomSearch(tr_X, tr_Y, poly)

    # Display optimum hyperparameters for SVC kernel
    print('Optimum hyperparameters for linear kernel: ')
    print(linSVC_param)
    print('Optimum hyperparameters for rbf kernel: ')
    print(rbfSVC_param)
    print('Optimum hyperparameters for polynomial kernel: ')
    print(polySVC_param)

    # printing validation accuracy score for each kernel
    print(lin_SVC.score(te_X, te_Y))
    print(rbf_SVC.score(te_X, te_Y))
    print(poly_SVC.score(te_X, te_Y))
    
    plt.figure(figsize=(9,18))

    # Cross validation with more iterations to get smoother mean test and train score curves, each time with 20% data randomly selected as a validation set.
    # SVC is more expensive so we do a lower number of CV iterations. 
    # cv = ShuffleSplit(n_splits=30, test_size=0.2, random_state=0)
    cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)

    # plot_learning_curve(estimator, title, te_X, te_Y, axes=axes, ylim=(0.8, 1.01), cv=cv, n_jobs=-1)
    axes = plt.subplot(311)
    title = r"Learning Curves (linear)"
    plot_learning_curve(lin_SVC, title, te_X, te_Y, axes=axes, ylim=(0.8, 1.01), cv=cv, n_jobs=-1)

    axes = plt.subplot(312)
    title = r"Learning Curves (rbf)"
    plot_learning_curve(rbf_SVC, title, te_X, te_Y, axes=axes, ylim=(0.7, 1.01), cv=cv, n_jobs=-1)

    axes = plt.subplot(313)
    title = r"Learning Curves (poly)"
    plot_learning_curve(poly_SVC, title, te_X, te_Y, axes=axes, ylim=(0.8, 1.01), cv=cv, n_jobs=-1)

    plt.show()
    
    return tr_X, tr_Y, rbf_SVC

####################### EVALUATION STARTS HERE #####################

def testResults(l1):
    A,b = get_data(l1)

    #similar preprocessing works for test data
    B = np.array([b, -(b - 1)]).T

    te_A = reshapeX(A)
    te_B = reshapeY(B)
    
    op_rbf_results = rbf_SVC.predict(te_A)

    cm = confusion_matrix(te_B, op_rbf_results)
    print(cm)
    plt.matshow(cm)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    print('\nConfusion matrix')
    plt.show()

    cm2 = confusion_matrix(te_B, op_rbf_results, normalize='all')
    print(cm2)
    plt.matshow(cm2)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    print('\nConfusion matrix(normalised)')
    plt.show()
    
    test_accuracy = accuracy_score(te_B, op_rbf_results)

    print(classification_report(te_B, op_rbf_results))
    print('Accuracy achieved:', test_accuracy, '\u2661''\u2661''\u2661')
    
    return test_accuracy