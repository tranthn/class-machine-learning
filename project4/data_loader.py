#!/usr/bin/env python3
import sys
import csv
import numpy as np
import pandas as pd

breast = '../data/breast-cancer-wisconsin.data'
glass = '../data/glass.data'
iris = '../data/iris.data'
soybean = '../data/soybean-small.data'
house = '../data/house-votes-84.data'

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
## class: 6 options
def get_glass_data():
    glass_fields = ['id','ri','na','mg', 'al','si','k','ca','ba','fe','class']
    std_fields = glass_fields[1:-1]

    gdf = read_csv(glass, glass_fields)
    gdf2 = gdf.copy().astype({'class': object}).drop(columns = 'id')
    gdf2 = standardize(gdf2, std_fields)

    data_sets = stratify_data(gdf2, 'class')

    return data_sets

########################################
## features are all continuous
## class: 3 options
def get_iris_data():
    iris_fields = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    std_fields = iris_fields[:-1]

    irdf = read_csv(iris, iris_fields)
    irdf = standardize(irdf, std_fields)
    data_sets = stratify_data(irdf, 'class')

    return data_sets

########################################
## attribute values are either multi-categorical or binary (assuming no missing)
## class: 4 options
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

########################################
## attribute values are either multi-categorical or binary (assuming no missing)
## ? = abstain, not missing values
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
    data_sets = stratify_data(housedf2, 'class')

    return data_sets
