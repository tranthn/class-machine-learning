#!/usr/bin/env python3
import sys
import csv
import numpy as np
import pandas as pd

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

        arr = np.histogram_bin_edges(values, bins='fd')
        df.loc[:, f] = pd.cut(x = values, bins = arr, include_lowest=True)

def split_tuning_training_data(df):
    tuning = df.sample(frac = 0.1)
    remainder = df.drop(tuning.index)
    train = remainder.sample(frac = 0.666667)
    test = remainder.drop(train.index)
    return {'tuning': tuning, 'train': train, 'test': test}

############### main ###############
## last column = malignancy (2 = benign, 4 = malignant)
## missing: 16 rows - missing 1 column value for bare_nuclei
## each column has domain: 1-10, need to be binned (except sample-code-number, class)
breast_fields = ['sample-code-number','clump-thickness','uniformity-of-cell-size','uniformity-of-cell-shape','marginal-adhesion','single-epithelial-cell-size','bare-nuclei','bland-chromatin','normal-nucleoli','mitoses','class']
bin_fields = breast_fields[1:-1]
bdf = read_csv(breast, breast_fields)
bdf2 = bdf[bdf['bare-nuclei'] != '?']
bdf2 = bdf2.astype({ 'bare-nuclei': int })
bdf3 = bdf2.copy()
bin_continuous(bdf3, bin_fields)
bdf4 = pd.get_dummies(bdf3)

# for c in bdf4.columns.tolist():
#     print(c)

print('data length', len(bdf4))
data_sets = split_tuning_training_data(bdf4)
print('data length, split', len(data_sets['tuning']), len(data_sets['train']), len(data_sets['test']))

print('---')

"""
## attribute values need values binned into ranges (except id, type)
## missing: none
glass_fields = ['id','ri','na','mg', 'al','si','k','ca','ba','fe','type']
bin_fields = glass_fields[1:-1]
gdf = read_csv(glass, glass_fields)
gdf2 = gdf.copy()
bin_continuous(gdf2, bin_fields)
gdf3 = pd.get_dummies(gdf2)
for c in gdf3.columns.tolist():
    print(c)
print('---')

## attribute values need values binned into ranges (except class)
## missing: none
iris_fields = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
bin_fields = iris_fields[:-1]
irdf = read_csv(iris, iris_fields)
irdf2 = irdf.copy()
bin_continuous(irdf2, bin_fields)
for c in irdf2.columns.tolist():
    print(c)
print('---')

## attribute values are either multi-categorical or binary (assuming no missing)
## missing: none
soybean_fields = ['date','plant-stand','precip','temp','hail','crop-hist','area-damaged','severity','seed-tmt','germination','plant-growth','leaves','leafspots-halo','leafspots-marg','leafspot-size','leaf-shread','leaf-malf','leaf-mild','stem','lodging','stem-cankers','canker-lesion','fruiting-bodies','external decay','mycelium','int-discolor','sclerotia','fruit-pods','fruit spots','seed','mold-growth','seed-discolor','seed-size','shriveling','roots']
soydf = read_csv(soybean, soybean_fields)
soydf2 = pd.get_dummies(soydf)
for c in soydf2.columns.tolist():
    print(c)
print('---')

## attribute values are either multi-categorical or binary (assuming no missing)
## ? = abstain, not missing values
## missing: none
house_fields = ['class', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']
housedf = read_csv(house, house_fields)
housedf2 = pd.get_dummies(housedf)
for c in housedf2.columns.tolist():
    print(c)
print('---')
"""