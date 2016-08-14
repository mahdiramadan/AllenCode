"""machine_learning.py by Mahdi Ramadan, 06-18-2016
This program will be used for machine learning fitting
and prediction
"""
import os
import pandas
import sys
# from image_processing import ImageProcessing as ip
from excel_processing import ExcelProcessing as ep
import pickle
from sklearn.svm import NuSVC
import cv2
import matplotlib.pyplot as plt
import numpy as np
import h5py
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import preprocessing
from multiprocessing import Process
from sklearn.cross_validation import train_test_split
from math import isnan
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer
import time
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier

def run_svm(final_data, y_train):

    # rows_n = len(input['feature_data'])
    # train = int(round(rows_n*0.8))
    # test = int(rows_n - train)
    #
    # X_train = input['feature_data'][0:train]
    # X_test = input['feature_data'][train:rows_n]
    #
    #
    # y_train = input['labels'][0:train]
    # y_test = input['labels'][train:rows_n]
    X_train, X_test, y_train, y_test = train_test_split(final_data, y_train, test_size=0.35, random_state= 32)

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [ 1e-3],
                         'C': [5, 10]}]

    scores = ['f1_weighted']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                           scoring='%s' % score, n_jobs= -1)
        clf.fit(X_train, y_train)

        joblib.dump(clf, 'clf.pkl')

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

    # # # # clf = RandomForestClassifier(verbose=3)
    # n_estimators = 10
    # clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='rbf', gamma = 0.001, C = 10),n_estimators = n_estimators), n_jobs = -1)
    #
    # clf.fit(X_train, y_train)
    #
    # y_true, y_pred = y_test, clf.predict(X_test)
    # print(classification_report(y_true, y_pred))
    #
    # # clf = joblib.load('filename.pk1')

def get_data(ep, lims_ID):
    hf = h5py.File(('data_' + str(lims_ID) + '.h5'), 'r')
    k = 0

    y_train = []
    dimension = 260*540
    wheel = joblib.load('dxds2.pkl')
    first_non_nan = next(x for x in wheel if not isnan(x))
    first_index = np.where(wheel == first_non_nan)[0]

    k = first_index[0]

    label = 'fidget'
    index = ep.get_labels().index(label) + 1
    fidget_vector = np.array(ep.get_per_frame_data()[index])

    label = 'walking'
    index = ep.get_labels().index(label) + 1
    walking = np.array(ep.get_per_frame_data()[index])

    label = 'running'
    index = ep.get_labels().index(label) + 1
    running = np.array(ep.get_per_frame_data()[index])

    movement_vector = []
    movement_vector.append([sum(x) for x in zip(walking, running)])

    group = hf.get('feature space')
    data = np.array(group.get('features'))[k:]

    for item in range(k, len(data)):
        if fidget_vector[k] == 1:
            y_train.append(0)
        elif movement_vector[0][k] == 1:
            y_train.append(1)
        else:
            y_train.append(2)

        k += 1
    return {'final_data': data, 'y_train': y_train}

def show_frame(frame):
    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    lims_ID = ['501560436', '501021421', '500860585']
    final_data = []
    y_train = []
    t= 0

    for itm in lims_ID:
        exl = ep("C:\Users\mahdir\Desktop\Mahdi files", itm)
        data = get_data(exl, itm)
        if t == 0:
            final_data = data['final_data']
            y_train= data['y_train']
        else:
            vector = data['final_data']
            final_data = np.vstack((final_data, vector))
            y_train = np.concatenate((y_train, data['y_train']))
        t += 1

    print('feature processing finished')
    p = Process(target = run_svm(final_data, y_train), args = (final_data, y_train))
    p.start()
    p.join()
