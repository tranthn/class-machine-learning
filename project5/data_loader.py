#!/usr/bin/env python3
import sys
import csv
import numpy as np
import pandas as pd

breast = '../data/breast-cancer-wisconsin.data'
glass = '../data/glass.data'
soybean = '../data/soybean-small.data'
abalone = '../data/abalone.data'
forestfires = '../data/forestfires.data'
machine = '../data/machine.data'

# generic csv reader wrapper function
def read_csv(file_path, fieldnames):
    try:
        df = pd.read_csv(file_path, names = fieldnames)

    except csv.Error as e:
        print("There was an error reading the file, exiting...")
        sys.exit('file read error: { }'.format(e))

    return df

# read csv data that contains header row already
def read_csv_with_header(file_path):
    try:
        df = pd.read_csv(file_path, header = 0)

    except csv.Error as e:
        print("There was an error reading the file, exiting...")
        sys.exit('file read error: { }'.format(e))

    return df

# will need to do stratification for the classfication data sets
# this will stratify based on indices
def stratify_data(df, label):
    df_orig = df.copy()
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

    sets = { 'tune': tune, 'folds': strats, 'all': df_orig }

    return sets

# folds for cross validation for regression data
# take out tuning first, then sort dataframe to split into folds
def stratify_regression_data(df, label):
    df_orig = df.copy()
    n = df.shape[0]
    fold = 5

    orig = df.copy()
    tune = df.sample(frac = 0.1, random_state=1)
    df = df.drop(tune.index)

    # sort remaining dataframe by our target class
    df = df.sort_values(by = [label])
    strats = []

    for i in range(fold):
        part = df.iloc[i::5]
        strats.append(part)

    sets = { 'tune': tune, 'folds': strats, 'all': df_orig }
    return sets

# convert continuous fields to standardized scale
def standardize(df, continuous_columns):
    for c in continuous_columns:
        df[c] = (df[c] - df[c].mean()) / df[c].std()

    return df

############### main data loading ###############
########################################
## features are continuous, each column has domain: 1-10
## class:  2 options (2 = benign, 4 = malignant) - remap to 0 = benign, 1 malignant
def get_breast_data():
    breast_fields = ['sample-code-number','clump-thickness','uniformity-of-cell-size',
                    'uniformity-of-cell-shape','marginal-adhesion','single-epithelial-cell-size',
                    'bare-nuclei','bland-chromatin','normal-nucleoli','mitoses','class']
    std_fields = breast_fields[1:-1]

    bdf = read_csv(breast, breast_fields)
    bdf = bdf.replace({'class': {4: 1, 2: 0}}).drop(columns = 'sample-code-number')
    bdf2 = bdf[bdf['bare-nuclei'] != '?']
    bdf3 = bdf2.copy().astype({ 'bare-nuclei': int })
    bdf4 = standardize(bdf3, std_fields)

    data_sets = stratify_data(bdf4, 'class')

    return data_sets

########################################
## features are all continuous 
## class: 6 options, 9 features
def get_glass_data():
    glass_fields = ['id','ri','na','mg', 'al','si','k','ca','ba','fe','class']
    std_fields = glass_fields[1:-1]

    gdf = read_csv(glass, glass_fields)
    gdf2 = gdf.copy().astype({'class': object}).drop(columns = 'id')
    gdf2 = standardize(gdf2, std_fields)

    data_sets = stratify_data(gdf2, 'class')
    return data_sets

########################################
## attribute values are either multi-categorical or binary (assuming no missing)
## class: 4 options, 35 columns (73 dummied)
def get_soy_data():
    soybean_fields = ['date','plant-stand','precip','temp','hail','crop-hist','area-damaged','severity',
                    'seed-tmt','germination','plant-growth','leaves','leafspots-halo','leafspots-marg',
                    'leafspot-size','leaf-shread','leaf-malf','leaf-mild','stem','lodging','stem-cankers',
                    'canker-lesion','fruiting-bodies','external decay','mycelium','int-discolor','sclerotia',
                    'fruit-pods','fruit spots','seed','mold-growth','seed-discolor','seed-size','shriveling','roots', 'class']
    bin_fields = soybean_fields[:-1]
    soydf = read_csv(soybean, soybean_fields)
    soydf = pd.get_dummies(soydf, columns = bin_fields)
    data_sets = stratify_data(soydf, 'class')

    return data_sets

####################### regression data sets #######################
## regression predictor: rings
def get_abalone_data():
    abalone_fields = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings']
    std_fields = abalone_fields[1:-1]
    abalone_df = read_csv(abalone, abalone_fields)
    abalone_df = pd.get_dummies(abalone_df, columns = ['sex'])
    abalone_df = standardize(abalone_df, std_fields)

    data_sets = stratify_regression_data(abalone_df, 'rings')
    return data_sets

########################################
## regression predictor: prp
## result indicator field [DO NOT USE IN MODEL]: erp
def get_machine_data():
    machine_fields = ['vendor_name', 'model_name', 'myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax', 'prp', 'erp']
    std_fields = machine_fields[2:-2]

    machine_df = read_csv(machine, machine_fields)
    machine_df = machine_df.drop(columns = ['erp', 'model_name'])
    machine_df2 = pd.get_dummies(machine_df, columns = ['vendor_name'])
    machine_df2 = standardize(machine_df2, std_fields)

    data_sets = stratify_regression_data(machine_df2, 'prp')
    return data_sets

########################################
## regression predictor: area [of fire]
def get_forest_fires_data():
    fire_fields = ['X', 'Y', 'month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']
    std_fields = fire_fields[4:-1]

    fire_df = read_csv_with_header(forestfires)
    fire_df = pd.get_dummies(fire_df, columns = ['month', 'day'])
    fire_df = standardize(fire_df, std_fields)

    data_sets = stratify_regression_data(fire_df, 'area')
    return data_sets