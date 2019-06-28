#!/bin/bash
rm build/output_*.txt
gcc src/simple_theta.c -o build/a.out
time build/a.out
build/theta_plot.py
