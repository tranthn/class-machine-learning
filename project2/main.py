#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import data_loader as dl
import knn

# field name that maps to the class column for data
# all data sets used here will be "class", but is externalized
# to make it easier if code needs to handle different class column names
class_label = 'class'

# pretty print the performance objects from cross validation
def print_helper_classifier(perf, fold):
    avg = 0
    for i, k in enumerate(perf):
        p = k['incorrect'] / k['total']
        pstr = "{:.0%}".format(p)
        print('fold #{0}, misclassified\t{1}'.format(i, pstr))
        avg += p

    print('------\navg. % misclassified:\t{:.0%}'.format(avg / fold))

def printer_helper_regressor(perf, fold):
    avg = 0
    for i, k in enumerate(perf):
        pstr = "{0:.3g}".format(k)
        print('fold #{0}, MSE\t{1}'.format(i, pstr))
        avg += k

    print('------\navg. MSE:\t{0:.3g}'.format(avg / fold))

# wrapper helpers to reduce code duplication for running knn
# optional execution of condensed or edit methods
def classification_helper(data, k, label = None, tuning = True, 
                            enn = False, cnn = False):

    if label == None:
        label = class_label

    f = 5 # fold-value
    tune = data['tune']

    knn_perf = []
    enn_perf = []
    cnn_perf = []

    for i in range(f):
        print('\n======== F O L D #{0} ========'.format(i))

        all_folds = data['folds'].copy()
        holdout = all_folds[i]
        folds = all_folds
        folds.pop(i)
        training = pd.concat(folds)

        # allow parameterized run with tuning or testing sets
        if (tuning):
            test = tune
        else:
            test = holdout

        # running classifer knn algorithms, edited and condensed methods are optional
        knn_results = knn.knn_classifier(training, test, label, k)
        knn_perf.append(knn_results)

        if (enn):
            print('\nedited knn')
            enn_set = knn.edited_knn(training, test, label)
            enn_results = knn.knn_classifier(enn_set, test, label, k)
            enn_perf.append(enn_results)

        if (cnn):
            print('\ncondensed knn')
            cnn_set = knn.condensed_knn(training, test, label)
            cnn_results = knn.knn_classifier(cnn_set, test, label, k)
            cnn_perf.append(cnn_results)

    print('-------')
    print('\n======== AVG. SUMMARY OF PERFORMANCE [KNN] ========')
    print_helper_classifier(knn_perf, f)

    print('\n======== AVG. SUMMARY OF PERFORMANCE [ENN] ========')
    print_helper_classifier(enn_perf, f)

    print('\n======== AVG. SUMMARY OF PERFORMANCE [CNN] ========')
    print_helper_classifier(cnn_perf, f)

## regression wrapper helper
def regression_helper(data, k, sigma, label = None, tuning = True, 
                        enn = False, cnn = False, threshold = None):

    if label == None:
        label = class_label
    
    f = 5 # fold-value
    tune = data['tune']

    knn_perf = []
    enn_perf = []
    cnn_perf = []

    print('\nT H R E S H O L D\t', threshold)
    print('---')
    for i in range(f):
        print('\n======== F O L D #{0} ========'.format(i))
        training = data['training'].copy()
        holdout = dl.sample_regression_data(training, label, i)
        training = training.drop(holdout.index)

        # allow parameterized run with tuning or testing sets
        if (tuning):
            test = tune
        else:
            test = holdout

        knn_results = knn.knn_regressor(training, test, label, k, sigma)
        knn_perf.append(knn_results)

        if (enn):
            print('\nedited knn')
            enn_set = knn.edited_knn(training, test, label, threshold)
            enn_results = knn.knn_regressor(enn_set, test, label, k, sigma)
            enn_perf.append(enn_results)

        if (cnn):
            print('\ncondensed knn')
            cnn_set = knn.condensed_knn(training, test, label, threshold)
            cnn_results = knn.knn_regressor(cnn_set, test, label, k, sigma)
            cnn_perf.append(cnn_results)
        
    print('----------')
    print('\n======== AVG. SUMMARY OF PERFORMANCE [KNN] ========')
    printer_helper_regressor(knn_perf, f)

    print('\n======== AVG. SUMMARY OF PERFORMANCE [ENN] ========')
    printer_helper_regressor(enn_perf, f)

    print('\n======== AVG. SUMMARY OF PERFORMANCE [CNN] ========')
    printer_helper_regressor(cnn_perf, f)

########################################
print('\n============== HOUSE DATA ============== ')
data = dl.get_house_data()
# classification_helper(data = data, k = 5, tuning = False, enn = True, cnn = True)

########################################
print('\n============== GLASS DATA ============== ')
data = dl.get_glass_data()
classification_helper(data = data, k = 11, tuning = False, enn = True, cnn = True)

########################################
print('\n============== SEGMENTATION DATA ============== ')
data = dl.get_segmentation_data()
# classification_helper(data = data, k = 13, label = 'CLASS', tuning = True, enn = True, cnn = True)

################# regression data sets #################
print('\n============== ABALONE DATA ============== ')
data = dl.get_abalone_data()
# regression_helper(data, k = 10, sigma = 2, label = 'rings', 
#                 threshold = 1, enn = True, cnn = True)

print('\n============== FOREST FIRE DATA ============== ')
data = dl.get_forest_fires_data()
# regression_helper(data, k = 20, sigma = 10, label = 'area', 
#                 threshold = 200, enn = True, cnn = True)

print('\n============== MACHINE DATA ============== ')
data = dl.get_machine_data()
regression_helper(data, k = 20, sigma = 5.3, label = 'prp', 
                threshold = 90, enn = True, cnn = True)