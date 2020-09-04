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

# def hot_code(data):

def equal_width_bin(values):
    N = 3
    m1 = min(values)
    m2 = max(values)
    interval = (m2 - m1) / N
    bins = [0] * (N+1)
    bins[0] = m1

    for i in range(1, N+1):
        bins[i] = bins[i - 1] + interval

    values2 = np.digitize(values, bins)
    print(values2)
    return values2

def bin_continuous(df, bin_fields):
    for f in bin_fields:
        values = df[f]
        binned_values = equal_width_bin(values)
        df[f] = binned_values
    return df

############### main ###############
## last column = malignancy (2 = benign, 4 = malignant)
## missing: 16 rows - missing 1 column value for bare_nuclei
## each column has domain: 1-10
breast_fields = ['sample_code_number','clump_thickness','uniformity_of_cell_size','uniformity_of_cell_shape','marginal_adhesion','single_epithelial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','class']
bdf = read_csv(breast, breast_fields)
bdf2 = bdf[bdf['bare_nuclei'] != '?']

## attribute values need values binned into ranges (except Type)
## missing: none
glass_fields = ['Id','RI','Na','Mg', 'Al','Si','K','Ca','Ba','Fe','Type']
bin_fields = glass_fields[:-1]
print(bin_fields)
gdf = read_csv(glass, glass_fields)
# print(gdf)
gdf2 = bin_continuous(gdf, bin_fields)
print(gdf2)

"""
## attribute values need values binned into ranges (except class)
## missing: none
iris_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
out = read_csv(iris, iris_fields)

## attribute values are either multi-categorical or binary (assuming no missing)
## missing: none
soybean_fields = ['date','plant-stand','precip','temp','hail','crop-hist','area-damaged','severity','seed-tmt','germination','plant-growth','leaves','leafspots-halo','leafspots-marg','leafspot-size','leaf-shread','leaf-malf','leaf-mild','stem','lodging','stem-cankers','canker-lesion','fruiting-bodies','external decay','mycelium','int-discolor','sclerotia','fruit-pods','fruit spots','seed','mold-growth','seed-discolor','seed-size','shriveling','roots']
out = read_csv(soybean, soybean_fields)

## ? = abstain, not missing values
## missing: none
house_fields = ['class', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras', 'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa']
out = read_csv(house, house_fields)
"""