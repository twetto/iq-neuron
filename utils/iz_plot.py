#!/usr/bin/python3

import glob
import numpy as np
import matplotlib.pyplot as plt

for filename in glob.glob("iz_output_*.txt"):
    with open(filename) as f:
        lines = []
        for line in f:
            lines.append(float(line.strip()))
    plt.figure()
    plt.title(filename)
    plt.plot(lines)
plt.show()

