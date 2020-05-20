#!/usr/bin/python3

import glob
import numpy as np
import matplotlib.pyplot as plt

for filename in glob.glob("ffcurve_*.txt"):
    with open(filename) as f:
        lines = []
        for line in f:
            lines.append(int(line.strip(), 10))
    x = np.linspace(0, 1, len(lines))
    plt.figure()
    plt.title(filename)
    plt.scatter(x, lines)
    #plt.plot(lines)
plt.show()

