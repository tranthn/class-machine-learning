#!/usr/bin/env python3
import sys
import csv
import numpy as np
import data_loader as dl
from simulator import TrackSimulator
from learner import QLearner
from value_iteration import ValueIteration

################################################################################

actions = [
    (-1,-1), (0,-1), (1,-1),
    (-1, 0), (0, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
]

vl_opts = [0, 1,  2, 3, 4, 5, -5, -4, -3, -2, -1]
vl_opts = [0, 1, -1] # shortened for speed/ease while testing

# tiny test track
track = dl.load_tinytrack()
simulator = TrackSimulator(track = track, min_velocity = min(vl_opts), max_velocity = max(vl_opts))
simulator.pretty_print()
learner = ValueIteration(env = simulator, vl_opts = vl_opts, actions = actions,
                        gamma = 1.0, epsilon = 0.1)

learner.value_iteration()

# l-track
# track = dl.load_l()
# simulator = TrackSimulator(track = track)
# simulator.initialize_track()
# simulator.pretty_print()

# r-track
# track = dl.load_r()
# simulator = TrackSimulator(track = track)
# simulator.initialize_track()
# simulator.pretty_print()

# o-track
# track = dl.load_o()
# simulator = TrackSimulator(track = track)
# simulator.initialize_track()
# simulator.pretty_print()