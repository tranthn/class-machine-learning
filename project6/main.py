#!/usr/bin/env python3
import sys
import csv
import numpy as np
import data_loader as dl
from simulator import TrackSimulator

################################################################################

# l-track
track = dl.load_l()
simulator = TrackSimulator(track = track)
simulator.initialize_track()
simulator.test_run()
simulator.pretty_print()

# r-track
track = dl.load_r()
simulator = TrackSimulator(track = track)
simulator.initialize_track()
simulator.pretty_print()

# o-track
track = dl.load_o()
simulator = TrackSimulator(track = track)
simulator.initialize_track()
simulator.pretty_print()