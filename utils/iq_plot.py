#!/usr/bin/python3.7

import glob
import numpy as np
import matplotlib.pyplot as plt

for filename in glob.glob("iq_output_*.txt"):
    with open(filename) as f:
        lines = []
        for line in f:
            lines.append(int(line.strip(), 10))
    plt.figure()
    plt.title(filename)
    plt.plot(lines)
plt.show()

