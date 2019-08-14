#!/usr/bin/bash

onewin=0
zerowin=0
noonewin=0
for i in {1..10000}
do
    ./main > output
    zeros=$(grep 0 output | wc -l)
    ones=$(grep 1 output | wc -l)
    if [ $ones -gt 100 ]
    then
        ((onewin++))
    elif [ $zeros -gt 100 ]
    then
        ((zerowin++))
    else
        ((noonewin++))
    fi
done

echo $noonewin $zerowin $onewin

