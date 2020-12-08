#!/usr/bin/env python3
import time
import data_loader as dl
from simulator import TrackSimulator
from learner import QLearner
from value_iteration import ValueIteration

def write_output_helper(file_prefix, input):
    # timestamp to append to file, month-day_hour
    timestamp = time.strftime('%m-%d_%H', time.localtime())
    filename = file_prefix + timestamp + '.out'
    with open(filename, 'w') as f:
        print(input, file = f)

################################################################################

actions = [
    (-1,-1), (0,-1), (1,-1),
    (-1, 0), (0, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
]

vl_opts = [0, 1,  2, 3, 4, 5, -5, -4, -3, -2, -1]
vl_opts = [0, 1, -1] # shortened for speed/ease while testing

# tiny test track
################################################################################
track = dl.load_tinytrack()
simulator = TrackSimulator(track = track, min_velocity = min(vl_opts), max_velocity = max(vl_opts), crash_restart = False)
learner = ValueIteration(env = simulator, vl_opts = vl_opts, actions = actions,
                        gamma = 1.0, epsilon = 0.1)

policy = learner.value_iteration(iterations = 50)
time_taken = simulator.run_trial(policy)
print('time: ', time_taken)

# l-track
################################################################################
# track = dl.load_l()
# simulator = TrackSimulator(track = track, min_velocity = min(vl_opts), max_velocity = max(vl_opts), crash_restart = False)
# simulator.pretty_print()
# learner = ValueIteration(env = simulator, vl_opts = vl_opts, actions = actions,
#                         gamma = 1.0, epsilon = 0.1)
# policy = learner.value_iteration()

# r-track
################################################################################
# track = dl.load_r()
# simulator = TrackSimulator(track = track, min_velocity = min(vl_opts), max_velocity = max(vl_opts), crash_restart = False)
# simulator.pretty_print()
# learner = ValueIteration(env = simulator, vl_opts = vl_opts, actions = actions,
#                         gamma = 1.0, epsilon = 0.1)
# policy = learner.value_iteration()

# o-track
################################################################################
# track = dl.load_o()
# simulator = TrackSimulator(track = track, min_velocity = min(vl_opts), max_velocity = max(vl_opts), crash_restart = False)
# simulator.pretty_print()
# learner = ValueIteration(env = simulator, vl_opts = vl_opts, actions = actions,
#                         gamma = 1.0, epsilon = 0.1)
# policy = learner.value_iteration()