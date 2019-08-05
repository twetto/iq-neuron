#!/usr/bin/bash

onewin=0
twowin=0
noonewin=0
for i in {1..10000}
do
    ./main > output
    zeros=$(grep 0 output | wc -l)
    ones=$(grep 1 output | wc -l)
    twos=$(grep 2 output | wc -l)
    if [ $ones -gt 300 ]
    then
        ((onewin++))
    elif [ $twos -gt 300 ]
    then
        ((twowin++))
    else
        ((noonewin++))
    fi
done

echo $noonewin $onewin $twowin

