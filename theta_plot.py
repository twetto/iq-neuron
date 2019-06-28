#!/usr/bin/python2.7
# simple plotting script for theta neuron.
# Chen-Fu Yeh, 2019.01.25

import glob
import numpy as np
import matplotlib.pyplot as plt

def read_integers(filename):
    with open(filename) as f:
        return map(int, f)

for filename in glob.glob("output_*.txt"):
    lines = read_integers(filename)
    plt.figure()
    plt.title(filename)
    plt.plot(lines)
plt.show()
