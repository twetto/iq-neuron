#!/bin/bash
for i in {0..128};
do
    echo -e "2000\n0 $i\n-1\n-1" > temp.txt
    ./IQIF < temp.txt | wc -l
done
