#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import data_loader
import winnow2 as win
import bayes

breast = './data/breast-cancer-wisconsin.data'
glass = './data/glass.data'
iris = './data/iris.data'
soybean = './data/soybean-small.data'
house = './data/house-votes-84.data'

# field name that maps to the class column for data
# all data sets used here will be "class", but is externalized
# to make it easier if code needs to handle different class column names
class_label = 'class'

############### main ###############
## breast data
print ('\n============== BREAST DATA ============== ')
data = data_loader.get_breast_data()

wts = win.build_classifier(df = data['train'], label = class_label)

win.test_model(data['test'], wts, label = class_label)

pt = bayes.build_probability_table(df = data['train'],
        label = class_label, m = 1, p = 0.01)

bayes.test_model(data['test'], pt, label = class_label)

########################################
## house data
print ('\n============== HOUSE DATA ============== ')
data = data_loader.get_house_data()

wts = win.build_classifier(df = data['train'], label = class_label)

win.test_model(data['test'], wts, label = class_label)

pt = bayes.build_probability_table(data['train'],
        label = class_label, m = 1, p = 0.01)
bayes.test_model(data['test'], pt, label = class_label)

########################################
## glass data
## data - returned as array with 2 sets: data[0] - winnow2, data[1] - bayes
print ('\n============== GLASS DATA ============== ')
data = data_loader.get_glass_data()

classifiers = win.build_classifier_multinomial(df = data[0]['train'], label = class_label)

win.test_model_multinomial(df = data[0]['test'], label = class_label, classifiers = classifiers)

pt = bayes.build_probability_table(data[1]['train'],
        label = class_label, m = 1, p = 0.01)
bayes.test_model(data[1]['test'], pt, label = class_label)

########################################
## iris data
## data - returned as array with 2 sets: data[0] - winnow2, data[1] - bayes
print ('\n============== IRIS DATA ============== ')
data = data_loader.get_iris_data()

classifiers = win.build_classifier_multinomial(df = data[0]['train'], label = class_label)

win.test_model_multinomial(df = data[0]['test'], label = class_label, classifiers = classifiers)

pt = bayes.build_probability_table(df = data[1]['train'],
        label = class_label, m = 1, p = 0.01)

bayes.test_model(data[1]['test'], pt, label = class_label)

########################################
## soy data
## data - returned as array with 2 sets: data[0] - winnow2, data[1] - bayes
print ('\n============== SOYBEAN DATA ============== ')
data = data_loader.get_soy_data()

classifiers = win.build_classifier_multinomial(df = data[0]['train'], label = class_label)

win.test_model_multinomial(df = data[0]['test'], label = class_label, classifiers = classifiers)

pt = bayes.build_probability_table(data[1]['train'],
        label = class_label, m = 1, p = 0.01)

bayes.test_model(data[1]['test'], pt, label = class_label)