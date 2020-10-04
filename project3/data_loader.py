#!/usr/bin/env python3
import sys
import csv
from fractions import Fraction
import numpy as np
import pandas as pd

### data-loader.py
# convenience class that handles:
# - data loading
# - data parsing
# - splitting data into train, tune, test sets

weather = './data/weather.data'
breast = '../data/breast-cancer-wisconsin.data'
car = './data/car.data'
segmentation = '../data/segmentation.data'
abalone = '../data/abalone.data'
machine = '../data/machine.data'
forestfires = '../data/forestfires.data'

## generic csv reader wrapper function
def read_csv(file_path, fieldnames):
    try:
        df = pd.read_csv(file_path, names = fieldnames)

    except csv.Error as e:
        print("There was an error reading the file, exiting...")
        sys.exit('file read error: { }'.format(e))

    return df

## read csv data that contains header row already
def read_csv_with_header(file_path):
    try:
        df = pd.read_csv(file_path, header = 0)

    except csv.Error as e:
        print("There was an error reading the file, exiting...")
        sys.exit('file read error: { }'.format(e))

    return df

## split data into 3 sets:
##  - tune: 10% of original
##  - remainder: to be used for cross validation
def split_tuning_data(df):
    orig = df.copy()
    tune = df.sample(frac = 0.1, random_state=1)
    df = df.drop(tune.index)
    return {'tune': tune, 'training': df, 'all': orig}

# will need to do stratification for the classfication data sets
# this will stratify based on indices
def stratify_data(df, label):
    n = df.shape[0]
    fold = 5

    # for any N that isn't by 5,
    # one fold must be smaller to hold the remainder
    f = n // fold
    last_fold = n // fold
    fold_size = (n - last_fold) // 4

    class_probabilities = []
    class_opts = df[label].value_counts()
    class_probabilities = class_opts.apply(lambda x: x / n)
    strats = []
    sample_df = df.copy()

    ## split off tuning set first
    ## we won't keep groupby frame since its multi-leveled
    ## we will grab its indices to reindex the original dataframe
    tune_df = df.groupby(label).apply(lambda x: x.sample(frac = 0.1, random_state = 1))
    part_idx = tune_df.index.get_level_values(1)
    tune = df.loc[part_idx, :]
    df = df.drop(part_idx)

    # remaining folds will be ~18% of original N of set
    for i in range(fold, 0, -1):
        sample_df = df.groupby(label).apply(lambda x: x.sample(frac = 1 / i, random_state = 1))
        part_idx = sample_df.index.get_level_values(1)
        part = df.loc[part_idx, :]
        strats.append(part)
        df = df.drop(part_idx)

    sets = { 'tune': tune, 'folds': strats }

    return sets

# folds for cross validation for regression data
# fold = int, representing which fold we're on
def sample_regression_data(df, sort_by, fold):
    df = df.sort_values(by = [sort_by])
    return df.iloc[fold::5]

############### main ###############
## dummy data from class lecture for weather
def get_weather():
    wea_df = read_csv_with_header(weather)
    wea_df = wea_df.replace({'class': {'p': 1, 'n': 0}})
    return wea_df

## each column has domain: 1-10, need to be binned (except sample-code-number, class)
## class:  2 options (2 = benign, 4 = malignant) - remap to 0 = benign, 1 malignant
def get_breast_data():
    breast_fields = ['sample-code-number','clump-thickness','uniformity-of-cell-size',
                    'uniformity-of-cell-shape','marginal-adhesion','single-epithelial-cell-size',
                    'bare-nuclei','bland-chromatin','normal-nucleoli','mitoses','class']
    bin_fields = breast_fields[1:-1]
    bdf = read_csv(breast, breast_fields)
    bdf = bdf.replace({'class': {4: 1, 2: 0}})
    bdf2 = bdf[bdf['bare-nuclei'] != '?']
    bdf3 = bdf2.copy().astype({ 'bare-nuclei': int })

    # drop unique id: sample-code-number, not needed
    bdf4 = pd.get_dummies(bdf3).drop(columns = 'sample-code-number')
    data_sets = stratify_data(bdf4, 'class')
    return data_sets

########################################s
# all fields are categorical
# class: 4 options
def get_car_data():
    car_fields = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    bin_fields = car_fields[:-1]
    cardf = read_csv(car, car_fields)
    data_sets = stratify_data(cardf, 'class')

    return data_sets

########################################
## class: 7 options
## bin fields: CLASS,REGION-CENTROID-COL,REGION-CENTROID-ROW,REGION-PIXEL-COUNT,SHORT-LINE-DENSITY-5,
# ## SHORT-LINE-DENSITY-2,VEDGE-MEAN,VEDGE-SD,HEDGE-MEAN,HEDGE-SD,INTENSITY-MEAN,RAWRED-MEAN,
# ## RAWBLUE-MEAN,RAWGREEN-MEAN,EXRED-MEAN,EXBLUE-MEAN,EXGREEN-MEAN,VALUE-MEAN,SATURATION-MEAN,HUE-MEAN
def get_segmentation_data():
    segmentation_df = read_csv_with_header(segmentation)
    data_sets = stratify_data(segmentation_df, 'CLASS')
    return data_sets

####################### regression data sets #######################
## regression predictor: rings
def get_abalone_data():
    abalone_fields = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings']
    abalone_df = read_csv(abalone, abalone_fields)

    abalone_df = pd.get_dummies(abalone_df, columns = ['sex'])
    data_sets = split_tuning_data(abalone_df)
    return data_sets

########################################
## regression predictor: prp
## result indicator field [DO NOT USE IN MODEL]: erp
def get_machine_data():
    machine_fields = ['vendor_name', 'model_name', 'myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax', 'prp', 'erp']
    bin_fields = machine_fields[:-1]
    machine_df = read_csv(machine, machine_fields)
    machine_df = machine_df.drop(columns = ['erp', 'model_name'])
    machine_df2 = pd.get_dummies(machine_df, columns = ['vendor_name'])
    data_sets = split_tuning_data(machine_df2)
    return data_sets

########################################
## regression predictor: area [of fire]
def get_forest_fires_data():
    fire_df = read_csv_with_header(forestfires)
    fire_df = pd.get_dummies(fire_df, columns = ['month', 'day'])
    data_sets = split_tuning_data(fire_df)
    return data_sets