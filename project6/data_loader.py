#!/usr/bin/env python3
import sys
import csv
import numpy as np

ltrack = './data/L-track.txt'
otrack = './data/O-track.txt'
rtrack = './data/R-track.txt'

"""
    – S – This square is on the starting line.
    – F – This square is on the finish line.
    – . – This square is open racetrack.
    – # – This square is off the racetrack (i.e., a wall).
"""

def load_file(filpepath):
    try:
        with open(filpepath) as f:
            # get dimensions for matrix and init
            lines = f.readlines()
            parts = lines[0].split(',')
            x = int(parts[0])
            y = int(parts[1])
            track = np.empty([x, y], str)

            line_idx = 0
            for line in lines[1:]:
                line = line.strip()
                parts = list(line)
                track[line_idx,] = parts
                line_idx += 1

    except IOError as err:
        print('There was an issue reading your file: {0}'.format(err))
        print('Exiting...')
        sys.exit(1)

    return track

def load_l():
    tracks = load_file(ltrack)
    return tracks

def load_o():
    tracks = load_file(otrack)
    return tracks

def load_r():
    tracks = load_file(rtrack)
    return tracks