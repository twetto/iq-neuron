#!/usr/bin/bash

latency=0

for i in {1..3000}
do
    ./main > output
    (( latency += $(sed -n 1p output) ))
done
echo $latency / 3000 | bc

