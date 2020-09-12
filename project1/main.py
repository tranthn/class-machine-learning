#!/usr/bin/env python3
import sys
import csv
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


############### main ###############
## breast data
data_sets = data_loader.get_breast_data()

# winnow2
# -----------------
train = data_sets['train']
tune = data_sets['tuning']
wts = win.build_classifier(df = train , label = 'class')
win.test_model(tune, wts, label = 'class')

# naive bayes
# -----------------
pt = bayes.build_probability_table(df = train, label = 'class')
bayes.test_model(tune, pt, label = 'class')

########################################
## data - returned as array with 2 sets:
#   data[0] - winnow2
#   data[1] - bayes
data = data_loader.get_glass_data()

# winnow2
# -----------------
classifiers = win.build_classifier_multinomial(df = data[0]['train'], label = 'class')
win.test_model_multinomial(df = data[0]['tuning'], label = 'class', classifiers = classifiers)

# naive bayes
# -----------------
pt = bayes.build_probability_table(data[1]['train'], label = 'class')

########################################
## data - returned as array with 2 sets:
#   data[0] - winnow2
#   data[1] - bayes
data = data_loader.get_iris_data()

# winnow2
# -----------------
train = data[0]['train']
tune = data[0]['tuning']
classifiers = win.build_classifier_multinomial(df = train, label = 'class')
win.test_model_multinomial(df = tune, label = 'class', classifiers = classifiers)

# naive bayes
# -----------------
train = data[1]['train']
tune = data[1]['tuning']
pt = bayes.build_probability_table(df = train, label = 'class')
bayes.test_model(tune, pt, label = 'class')

########################################
## data - returned as array with 2 sets:
#   data[0] - winnow2
#   data[1] - bayes
data = data_loader.get_soy_data()

# winnow2
# -----------------
classifiers = win.build_classifier_multinomial(df = data[0]['train'], label = 'class')
win.test_model_multinomial(df = data[0]['tuning'], label = 'class', classifiers = classifiers)

# naive bayes
# -----------------
pt = bayes.build_probability_table(data[1]['train'], label = 'class')

########################################
## house data
data = data_loader.get_house_data()

# winnow2
# -----------------
# wts = win.build_classifier(df = data[0]['train'], label = 'class')
# win.test_model(data[0]['tuning'], wts)

# # naive bayes
# # -----------------
# pt = bayes.build_probability_table(data[1]['train'], label = 'class')