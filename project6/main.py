#!/usr/bin/env python3
import time
import data_loader as dl
from simulator import TrackSimulator
from learner import QLearner
from value_iteration import ValueIteration

def write_output_helper(file_prefix, input):
    # timestamp to append to file, month-day_hour
    timestamp = time.strftime('%m-%d_%H', time.localtime())
    filename = file_prefix + '_' + timestamp + '.out'
    with open(filename, 'w') as f:
        print(input, file = f)

def trial_helper(simulator, learner, iterations, trial_runs, trackname):
    print('\nstart of training, for {0} total iterations'.format(iterations))
    st = time.time()
    policy = learner.value_iteration(iterations = iterations)
    print('\npolicy finalized after {:.2f}s'.format(time.time() - st))
    print('\n# time trials to run: ', trial_runs)
    print()
    for i in range(trial_runs):
        start_time = time.time()
        print('\ttrial', i)
        steps = simulator.run_trial(policy)

        # store policy for later use if needed
        write_output_helper(trackname, policy)
        print('\t# moves taken: ', steps)
        elapsed = time.time() - start_time
        print('\ttrial runtime:\t{:.2f}s'.format(elapsed))
        print()

    print('=' * 100)

################################################################################

actions = [
    (-1,-1), (0,-1), (1,-1),
    (-1, 0), (0, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
]

vl_opts = [0, 1,  2, 3, 4, 5, -5, -4, -3, -2, -1]
# vl_opts = [0, 1, -1] # shortened for speed/ease while testing

# tiny test track
################################################################################
track = dl.load_tinytrack()
simulator = TrackSimulator(track = track, min_velocity = min(vl_opts), max_velocity = max(vl_opts), crash_restart = False)
learner = ValueIteration(env = simulator, vl_opts = vl_opts, actions = actions,
                        gamma = 1.0, epsilon = 0.01)

simulator.pretty_print()
# trial_helper(simulator, learner, 30, 10, 'tinytrack')

# l-track
################################################################################
track = dl.load_l()
print()
print('=' * 100)
print('RUNNING L TRACK')
simulator = TrackSimulator(track = track, min_velocity = min(vl_opts), max_velocity = max(vl_opts), crash_restart = False)
learner = ValueIteration(env = simulator, vl_opts = vl_opts, actions = actions,
                        gamma = 0.9, epsilon = 0.001)

simulator.pretty_print()
trial_helper(simulator, learner, 50, 10, 'L-track')

# r-track
################################################################################
track = dl.load_r()
print()
print('=' * 100)
print('RUNNING R TRACK')
simulator = TrackSimulator(track = track, min_velocity = min(vl_opts), max_velocity = max(vl_opts), crash_restart = False)
learner = ValueIteration(env = simulator, vl_opts = vl_opts, actions = actions,
                        gamma = 0.9, epsilon = 0.001)

simulator.pretty_print()
# trial_helper(simulator, learner, 50, 10, 'R-track')

# o-track
################################################################################
track = dl.load_o()
print()
print('=' * 100)
print('RUNNING O TRACK')
simulator = TrackSimulator(track = track, min_velocity = min(vl_opts), max_velocity = max(vl_opts), crash_restart = False)
learner = ValueIteration(env = simulator, vl_opts = vl_opts, actions = actions,
                        gamma = 0.9, epsilon = 0.001)

simulator.pretty_print()
# trial_helper(simulator, learner, 50, 10, 'O-track')