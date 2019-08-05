#!/usr/bin/bash

zeroLatency=0
oneLatency=0
twoLatency=0

for i in {1..3000}
do
    ./main > output
    (( zeroLatency += $(sed -n 1p output) ))
    (( oneLatency += $(sed -n 2p output) ))
    (( twoLatency += $(sed -n 3p output) ))
done
echo $zeroLatency / 3000 | bc
echo $oneLatency / 3000 | bc
echo $twoLatency / 3000 | bc

