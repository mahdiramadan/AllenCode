"""machine_learning.py by Mahdi Ramadan, 06-18-2016
This program will be used for machine learning fitting
and prediction
"""
import os
import pandas
import sys
from image_processing import ImageProcessing as ip
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


def run_svm(input):

    rows_n = len(input['feature_data'])
    train = int(round(rows_n*0.8))
    test = int(rows_n - train)

    X_train = input['feature_data'][0:train]
    X_test = input['feature_data'][train:rows_n]


    y_train = input['labels'][0:train]
    y_test = input['labels'][train:rows_n]

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [ 1, 1e-3, 1e-4],
                         'C': [ 0.1, 1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [ 0.1, 1, 10, 1000]}]

    scores = ['accuracy', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                           scoring='%s' % score, n_jobs= -1)
        clf.fit(X_train, y_train)

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

    # # # clf = RandomForestClassifier(verbose=3)
    # # clf = SVC(kernel='linear', C = 0.1, verbose = 2)
    #
    # clf = neighbors.KNeighborsClassifier(2)
    # clf.fit(X_train, y_train)
    #
    # joblib.dump(clf, 'clf.pkl')
    #
    # y_true, y_pred = y_test, clf.predict(X_test)
    # print(classification_report(y_true, y_pred))

    # # clf = joblib.load('filename.pk1')

def get_data(ep):

    hf = h5py.File('data.h5', 'r')
    final_data = []
    y_train = []
    dimension = 260*540
    wheel = pickle.load(open('wheel.pickle', 'rb'))
    wheel = (wheel - np.min(wheel)) / (np.max(wheel) - np.min(wheel) + 10 ** -10)

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

    count = 0
    k = 0
    t= 0

    hsv = np.zeros((260, 540, 3))
    hsv[..., 1] = 255


    for item in range(1, 21):
        group = hf.get('first ' + str(item) + '000 frames')
        # optical.append(np.array(group.get('optical')))
        # angles.append(np.array(group.get('angles')))
        frames = np.array(group.get('frames'))
        opticals = np.array(group.get('optical'))
        angles = np.array(group.get('angles'))
        count += 1

        print ('first ' + str(count) + '000 frames')

        # width = len(frames[0][0])
        # height = len(frames[0])
        # dim = height / 2 * (width / 2)

        for f in range(len(frames)):


            # hog = self.hog(frames[f])
            # hog =
            #  np.reshape(hog, (1, len(hog)))
            # hog = (hog - hog.min())/(hog.max() - hog.min() + 10**-1)

            # section_2 = frames[f][0:height / 3, width / 4:width]
            # height2= len(section_2)
            # width2= len(section_2[0])
            #
            # section_3 = frames[f][height/3: 2*height/3, 0: width/2]
            # height3 = len(section_3)
            # width3 = len(section_3[0])
            #
            # section_4 = frames[f][height/3: height, width/2: width]
            # height4= len(section_4)
            # width4= len(section_4[0])

            # hist_1, bin_1 = np.histogram(np.reshape((opticals[f][height / 2: height, 0: width / 2]), (1, dim)), 20)
            # b_width = 0.7 * (bin_1[1] - bin_1[0])
            # center = (bin_1[:-1] + bin_1[1:]) / 2
            # plt.bar(center, hist_1, align='center', width=b_width)
            # plt.show()
            #
            # hist_1, bin_1 = np.histogram(np.reshape((opticals[f][0:height / 2, width / 2:width]), (1, dim)), 20)
            # b_width = 0.7 * (bin_1[1] - bin_1[0])
            # center = (bin_1[:-1] + bin_1[1:]) / 2
            # plt.bar(center, hist_1, align='center', width= b_width)
            # plt.show()
            #
            # hist_1, bin_1 = np.histogram(np.reshape((opticals[f][height / 2:height, width / 2: width]), (1, dim)), 20)
            # b_width = 0.7 * (bin_1[1] - bin_1[0])
            # center = (bin_1[:-1] + bin_1[1:]) / 2
            # plt.bar(center, hist_1, align='center', width= b_width)
            # plt.show()
            #
            #
            # optical = np.reshape(optical, (1, len(optical)))
            # optical = (optical - optical.min()) / (optical.max() - optical.min() + 10**-10)
            #
            # angle = self.hog(angles[f])
            # angle = np.reshape(angle, (1, len(angle)))
            # angle = (angle - angle.min()) / (angle.max() - angle.min()+ 10**-10)
            #
            # vector = np.int16(np.hstack(( optical, angle)))

            if fidget_vector[k] == 1 or movement_vector[0][k] == 1:
                # hsv[..., 0] = angles[k]
                # hsv[..., 2] = cv2.normalize(opticals[k], None, 0, 255, cv2.NORM_MINMAX)
                # hsv = np.float32(hsv)
                # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                #
                # r = np.int64(rgb[:,:,0])
                # # x = x[np.nonzero(x)]
                # b = np.int64(rgb[:,:,1])
                # g = np.int64(rgb[:,:,2])
                #
                # frame_data = []
                # vector = []
                optic1 = []
                optic2 = []
                # optic3= []

                # for (x,y,window) in self.sliding_window(frames[k][0:240, :], 60, (60,60)):
                #     hist, bin = np.histogram(window, 10)
                #     center = (bin[:-1] + bin[1:]) / 2
                #     hist_x = np.multiply(center, hist)
                #     hist_x = (hist_x- np.min(hist_x)) / (np.max(hist_x) - np.min(hist_x) + 10 ** -10)
                #     frame_data = np.concatenate((frame_data, hist_x))

                for (x, y, window) in sliding_window(opticals[f][0:240, :], 60, (60, 60)):
                    hist, bin = np.histogram(window, 10)
                    center = (bin[:-1] + bin[1:]) / 2
                    hist_x = np.multiply(center, hist)
                    hist_x = preprocessing.MinMaxScaler((-1,1)).fit(hist_x).transform(hist_x)
                    optic1 = np.concatenate((optic1, hist_x))

                for (x, y, window) in sliding_window(angles[f][0:240, :], 60, (60, 60)):
                    hist, bin = np.histogram(window, 10)
                    center = (bin[:-1] + bin[1:]) / 2
                    hist_x = np.multiply(center, hist)
                    hist_x = preprocessing.MinMaxScaler((-1,1)).fit(hist_x).transform(hist_x)
                    optic2 = np.concatenate((optic2, hist_x))

                # for (x, y, window) in self.sliding_window(g[0:240, :], 60, (60, 60)):
                #     hist, bin = np.histogram(window, 10)
                #     center = (bin[:-1] + bin[1:]) / 2
                #     hist_x = np.multiply(center, hist)
                #     hist_x = (hist_x - np.min(hist_x)) / (np.max(hist_x) - np.min(hist_x) + 10 ** -10)
                #     optic3 = np.concatenate((optic3, hist_x))


                if t == 0:
                    final_data = np.concatenate((optic1, optic2))

                else:
                    vector = np.concatenate((optic1, optic2))
                    final_data = np.vstack((final_data, vector))

                if fidget_vector[k] == 1:
                    y_train.append(0)
                else:
                    y_train.append(1)

                t += 1


            k += 1

    print( 'feature processing finished')

    return {'feature_data': final_data, 'labels': y_train}

def hog(img):

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    return mag

def show_frame(frame):
    cv2.imshow('image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


if __name__ == '__main__':
    ep = ep("C:\Users\mahdir\Desktop\Mahdi files", "501560436")
    data = get_data(ep)
    p = Process(target = run_svm(data), args = data)
    p.start()
    p.join()
