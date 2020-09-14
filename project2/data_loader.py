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
# - binning continuous data into discrete bins
# - one-hot encoding
# - splitting data into train, tune, test sets

glass = '../data/glass.data'
house = '../data/house-votes-84.data'
abalone = './data/abalone.data'
forestfires = './data/forestfires.data'
machine = './data/machine.data'
segmentation = './data/segmentation.data'

## generic csv reader wrapper function
def read_csv(file_path, fieldnames):
    try:
        df = pd.read_csv(file_path, names = fieldnames)

    except csv.Error as e:
        print("There was an error reading the file, exiting...")
        sys.exit('file read error: { }'.format(e))

    return df

def read_csv_with_header(file_path):
    try:
        df = pd.read_csv(file_path, header = 0)

    except csv.Error as e:
        print("There was an error reading the file, exiting...")
        sys.exit('file read error: { }'.format(e))

    return df

## data pre-processing
## for multi-value categorical: one-hot coding to turn each potential category value to its own boolean column
## only one of the category options will be "hot", i.e = 1, in a given row
## for values with continuous ranges - bin them into ranges, then do one-hot coding
def bin_continuous(df, bin_fields):
    for f in bin_fields:
        values = df[f]
        m1 = min(values)
        m2 = max(values)

        arr = np.histogram_bin_edges(values, bins='auto')
        df.loc[:, f] = pd.cut(x = values, bins = arr, include_lowest=True)

## split data into 3 sets:
##  - tune: 10% of original
##  - train: 67% of remainder
##  - test: remainder of remainder
def split_tuning_training_data(df):
    orig = df.copy()
    tune = df.sample(frac = 0.1, random_state=1)
    df = df.drop(tune.index)
    train = df.sample(frac = 0.67, random_state=1)
    test = df.drop(train.index)
    return {'tune': tune, 'train': train, 'test': test, 'all': orig}

########################################
## attribute values are either multi-categorical or binary (assuming no missing)
## ? = abstain, not missing values
## missing: none
## class: 2 options
def get_house_data():
    house_fields = ['class', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution',
                    'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban',
                    'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback', 
                    'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']
    bin_fields = house_fields[1:]
    housedf = read_csv(house, house_fields)

    # manually replace democrat and republican for boolean fields
    # using default pandas replace wasn't working on object type
    housedf['class'] = housedf['class'].str.strip().replace('democrat', '0')
    housedf['class'] = housedf['class'].str.strip().replace('republican', '1')
    housedf['class'] = pd.to_numeric(housedf['class'])

    housedf2 = pd.get_dummies(housedf, columns = bin_fields)
    data_sets = split_tuning_training_data(housedf2)
    return data_sets

########################################
def get_glass_data():
    glass_fields = ['id','ri','na','mg', 'al','si','k','ca','ba','fe','class']
    bin_fields = glass_fields[1:-1]
    gdf = read_csv(glass, glass_fields)
    gdf2 = gdf.copy().astype({'class': object})
    bin_continuous(gdf2, bin_fields)
    gdf3 = pd.get_dummies(gdf2).drop(columns = 'id')
    gdf3_bayes = pd.get_dummies(gdf2, columns = bin_fields).drop(columns = 'id')
    data_sets = split_tuning_training_data(gdf3)
    return data_sets

############### main ###############
def get_abalone_data():
    abalone_fields = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings']
    abalone_df = read_csv(abalone, abalone_fields)
    data_sets = split_tuning_training_data(abalone_df)
    return data_sets

########################################
## HAS HEADER DATA BUILT IN ALREADY
def get_forest_fires_data():
    forestfires_fields = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    bin_fields = forestfires_fields[:-1]
    fire_df = read_csv(forestfires, forestfires_fields)
    data_sets = split_tuning_training_data(fire_df)
    return data_sets

########################################
def get_machine_data():
    machine_fields = ['vendor_name', 'model_name', 'myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax', 'prp', 'erp']
    bin_fields = machine_fields[:-1]
    machine_df = read_csv(machine, machine_fields)
    machine_df2 = pd.get_dummies(machine_df)
    data_sets = split_tuning_training_data(machine_df2)
    return data_sets

########################################
## HAS HEADER DATA BUILT IN ALREADY
def get_segmentation_data():
    segmentation_df = read_csv_with_header(segmentation)
    data_sets = split_tuning_training_data(segmentation_df)
    return data_sets