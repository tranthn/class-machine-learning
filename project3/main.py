#!/usr/bin/env python3
import sys
import time
import numpy as np
import pandas as pd
import data_loader as dl
from id3 import ID3Tree
from regression import RegressionTree

# field name that maps to the class column for data
# all data sets used here will be "class", but is externalized
# to make it easier if code needs to handle different class column names
class_label = 'class'

# pretty print the performance objects from cross validation
def print_helper_classifier(perf, fold):
    avg = 0
    for i, k in enumerate(perf):
        pstr = "{:.0%}".format(k)
        print('fold #{0}, accuracy\t{1}'.format(i, pstr))
        avg += k

    print('------\navg. % accuracy:\t{:.0%}'.format(avg / fold))

def printer_helper_regressor(perf, fold):
    avg = 0
    for i, k in enumerate(perf):
        pstr = "{0:.5g}".format(k)
        print('fold #{0}, MSE\t{1}'.format(i, pstr))
        avg += k

    print('------\navg. MSE:\t{0:.5g}'.format(avg / fold))

# wrapper helpers to reduce code duplication for running knn
# optional execution of condensed or edit methods
def classification_helper(data, label = None):
    if label == None:
        label = class_label

    f = 5 # fold-value
    perf = []
    elapsed_time = 0

    # get starting attrs for building tree
    tune = data['tune']
    attrs = tune.drop(columns = [label]).columns.values

    for i in range(f):
        print('\n========= F O L D #{0} ========='.format(i))
        folds = data['folds'].copy()
        holdout = folds[i]
        folds.pop(i) # remove holdout fold
        training = pd.concat(folds) # concat remaining folds to create training set

        # build the tree model
        id3_tree = ID3Tree(validation_set = tune)
        trained_tree = id3_tree.id3_tree(df = training, label = label, tree = None, features = attrs)
        print('# tree nodes:\t\t{0}'.format(id3_tree.num_nodes))

        # time the run for testing tree
        start_time = time.time()
        result = id3_tree.test_tree(trained_tree, holdout, label)
        print('tree accuracy:\t\t{:.0%}'.format(result))
        elapsed = time.time() - start_time
        elapsed_time += elapsed
        print("execution time:\t\t{:.2f}s".format(elapsed))
        perf.append(result)

    print('------------')
    print('\n====== AVG. SUMMARY OF PERFORMANCE ======')
    print_helper_classifier(perf, f)
    print('avg. runtime:\t\t{:.2f}s'.format(elapsed_time / 5))

## regression wrapper helper
def regression_helper(data, label = None, threshold = None):
    if label == None:
        label = class_label

    f = 5 # fold-value
    tune = data['tune']
    perf = []
    elapsed_time = 0

    print('\nTHRESHOLD\t', threshold)
    for i in range(f):
        print('\n========= F O L D #{0} ========='.format(i))
        folds = data['folds'].copy()
        holdout = folds[i]
        folds.pop(i) # remove holdout fold
        training = pd.concat(folds) # concat remaining folds to create training set

        # build the tree model
        reg = RegressionTree(validation_set = tune, threshold = threshold, node_min = 25)
        tree = reg.reg_tree(df = training, label = label, tree = None, prior_value = None)
        print('# tree nodes:\t\t{0}'.format(reg.num_nodes))

        # time the run for testing tree
        start_time = time.time()
        result = reg.test_tree(tree, holdout, label)
        print('tree accuracy:\t\t{:.0%}'.format(result))
        elapsed = time.time() - start_time
        elapsed_time += elapsed
        print("execution time:\t\t{:.2f}s".format(elapsed))
        perf.append(result)
        
    print('------------')
    print('\n====== AVG. SUMMARY OF PERFORMANCE ======')
    printer_helper_regressor(perf, f)
    print('avg. runtime:\t\t{:.2f}s'.format(elapsed_time / 5))

################ classification data sets ################

print('\n================== BREAST DATA ================== ')
data = dl.get_breast_data()
classification_helper(data, label = 'class')

print('\n==================== CAR DATA ==================== ')
data = dl.get_car_data()
classification_helper(data, label = 'class')

print('\n================== SEGMENTATION DATA ================== ')
data = dl.get_segmentation_data()
label = 'CLASS'
classification_helper(data, label)

################# regression data sets #################
print('\n================== ABALONE DATA ================== ')
data = dl.get_abalone_data()
label = 'rings'
threshold = 9.8

# regression_helper(data, label, threshold = 0)
# print()
# regression_helper(data, label, threshold = threshold)

print('\n================== MACHINE DATA ================== ')
data = dl.get_machine_data()
label = 'prp'
threshold = 4150

# regression_helper(data, label, threshold = 0)
# print()
# regression_helper(data, label, threshold = threshold)

print('\n================== FOREST FIRE DATA ================== ')
data = dl.get_forest_fires_data()
label = 'area'
threshold = 10680

# regression_helper(data, label, threshold = 0)
# print()
# regression_helper(data, label, threshold = threshold)