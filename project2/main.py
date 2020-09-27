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

# helpers to reduce code duplication for running knn
# optional exeuction of condensed or edit methods
def classification_helper(data, k, label = None, tuning = True, 
                            enn = False, cnn = False):

    if label == None:
        label = class_label

    f = 5 # fold-value
    tune = data['tune']

    for i in range(f):
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
        knn_model = knn.knn_classifier(training, test, label, k)

        if (enn):            
            enn_set = knn.edited_knn(training, test, label)
            enn_model = knn.knn_classifier(enn_set, test, label, k)

        if (cnn):
            cnn_set = knn.condensed_knn(training, test, label)
            cnn_model = knn.knn_classifier(cnn_set, test, label, k)

def regression_helper(data, k, sigma, label = None, tuning = True, 
                        enn = False, cnn = False, threshold = None):

    if label == None:
        label = class_label
    
    f = 5 # fold-value
    tune = data['tune']

    print('threshold\t', threshold)
    print('---')
    for i in range(f):
        print('fold\t', i)
        training = data['training'].copy()
        holdout = dl.sample_regression_data(training, label, i)
        training = training.drop(holdout.index)

        # allow parameterized run with tuning or testing sets
        if (tuning):
            test = tune
        else:
            test = holdout

        # knn.knn_regressor(training, test, label, k, sigma)

        if (enn):
            enn_set = knn.edited_knn(training, test, label, threshold)
            enn_model = knn.knn_regressor(enn_set, test, label, k, sigma)

        if (cnn):
            cnn_set = knn.condensed_knn(training, test, label, threshold)
            cnn_model = knn.knn_regressor(cnn_set, test, label, k, sigma)
        
        print('----------')

########################################
print('\n============== HOUSE DATA ============== ')
data = dl.get_house_data()
# classification_helper(data = data, k = 5, tuning = True, enn = True, cnn = True)

########################################
print('\n============== GLASS DATA ============== ')
data = dl.get_glass_data()
# classification_helper(data = data, k = 5)

########################################
print('\n============== SEGMENTATION DATA ============== ')
data = dl.get_segmentation_data()
class_label = 'CLASS'
# classification_helper(data = data, k = 5, label = class_label)

################# regression data sets #################
print('\n============== ABALONE DATA ============== ')
data = dl.get_abalone_data()

print('\n============== FOREST FIRES DATA ============== ')
data = dl.get_forest_fires_data()
# print(data)

print('\n============== MACHINE DATA ============== ')
data = dl.get_machine_data()
regression_helper(data, k = 20, sigma = 5.3, label = 'prp', threshold = 90, enn = True)