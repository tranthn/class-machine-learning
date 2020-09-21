#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd

### for a given point p1, look at # k nearest points to p1
### distance/class of near neighbors and majority class
### we will label the point p1 with said class

### picking k
# k = odd, for 2-class problem
# k = not multiple of # classes

def euclidean_dist(v1, v2):
    square = lambda x : x ** 2
    v = np.subtract(v1, v2)
    dist = np.sum(np.array([square(vi) for vi in v]))
    print(dist)