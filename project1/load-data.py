#!/usr/bin/env python3
import sys
import csv
from fractions import Fraction
import numpy as np
import pandas as pd
import winnow2 as win
import bayes

breast = './data/breast-cancer-wisconsin.data'
glass = './data/glass.data'
iris = './data/iris.data'
soybean = './data/soybean-small.data'
house = './data/house-votes-84.data'

## generic csv reader wrapper function
def read_csv(file_path, fieldnames):
    try:
        df = pd.read_csv(file_path, names = fieldnames)

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

def split_tuning_training_data(df):
    tuning = df.sample(frac = 0.1)
    df = df.drop(tuning.index)
    train = df.sample(frac = 0.6)
    test = df.drop(train.index)
    return {'tuning': tuning, 'train': train, 'test': test}

############### main ###############
## each column has domain: 1-10, need to be binned (except sample-code-number, class)
## missing: 16 rows - missing 1 column value for bare_nuclei
## class:  2 options (2 = benign, 4 = malignant) - remap to 0 = benign, 1 malignant
breast_fields = ['sample-code-number','clump-thickness','uniformity-of-cell-size','uniformity-of-cell-shape','marginal-adhesion','single-epithelial-cell-size','bare-nuclei','bland-chromatin','normal-nucleoli','mitoses','class']
bin_fields = breast_fields[1:-1]
bdf = read_csv(breast, breast_fields)
bdf = bdf.replace({'class': {4: 1, 2: 0}})
bdf2 = bdf[bdf['bare-nuclei'] != '?']
bdf3 = bdf2.copy().astype({ 'bare-nuclei': int })
bin_continuous(bdf3, bin_fields)

# drop sample-code-number since it not needed for learning model
bdf4 = pd.get_dummies(bdf3).drop(columns = 'sample-code-number')
data_sets = split_tuning_training_data(bdf4)

# winnow2
# -----------------
wts = win.build_table(df = data_sets['train'], label = 'class')
win.test_model(data_sets['tuning'], wts)

# # naive bayes
# # -----------------
pt = bayes.build_probability_table(data_sets['train'])

########################################
## attribute values need values binned into ranges (except id, class)
## missing: none
## class: 6 options
glass_fields = ['id','ri','na','mg', 'al','si','k','ca','ba','fe','class']
bin_fields = glass_fields[1:-1]
gdf = read_csv(glass, glass_fields)
gdf2 = gdf.copy().astype({'class': object})
bin_continuous(gdf2, bin_fields)
gdf3 = pd.get_dummies(gdf2).drop(columns = 'id')
data_sets = split_tuning_training_data(gdf3)
print(data_sets['tuning'])

multi_tables = win.build_table_multinomial(df = gdf3, label = 'class')
print(len(multi_tables))

########################################
## attribute values need values binned into ranges (except class)
## missing: none
## class: 3 options
iris_fields = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
bin_fields = iris_fields[:-1]
irdf = read_csv(iris, iris_fields)
irdf2 = irdf.copy()
bin_continuous(irdf2, bin_fields)
irdf3 = pd.get_dummies(irdf2, columns = ['class'])
data_sets = split_tuning_training_data(irdf2)
# print(data_sets['tuning'])

########################################
## attribute values are either multi-categorical or binary (assuming no missing)
## missing: none
## class: 4 options
soybean_fields = ['date','plant-stand','precip','temp','hail','crop-hist','area-damaged','severity','seed-tmt','germination','plant-growth','leaves','leafspots-halo','leafspots-marg','leafspot-size','leaf-shread','leaf-malf','leaf-mild','stem','lodging','stem-cankers','canker-lesion','fruiting-bodies','external decay','mycelium','int-discolor','sclerotia','fruit-pods','fruit spots','seed','mold-growth','seed-discolor','seed-size','shriveling','roots', 'class']
soydf = read_csv(soybean, soybean_fields)
soydf2 = pd.get_dummies(soydf)
data_sets = split_tuning_training_data(soydf2)
# print(data_sets['tuning'])

########################################
## attribute values are either multi-categorical or binary (assuming no missing)
## ? = abstain, not missing values
## missing: none
## class: 2 options
house_fields = ['class', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']
housedf = read_csv(house, house_fields)
housedf2 = pd.get_dummies(housedf)
data_sets = split_tuning_training_data(housedf)
# print(data_sets['tuning'])