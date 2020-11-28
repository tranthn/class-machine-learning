#!/usr/bin/env python3
import sys
import csv
import numpy as np
import data_loader as dl
from simulator import TrackSimulator

################################################################################

ltrack = dl.load_l()
# print(ltrack.shape)
simulator = TrackSimulator(track = ltrack)
simulator.initialize_track()
simulator.pretty_print()