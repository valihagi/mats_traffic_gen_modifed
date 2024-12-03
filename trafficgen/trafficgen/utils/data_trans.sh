#!/usr/bin/env bash


for i in 0 1 2 3 4 5 6 7 8 9; do
     mkdir $1/raw_${i}
     mv $1/training_20s.tfrecord-00${i}* $1/raw_${i}/
done

for i in 0 1 2 3 4 5 6 7 8 9; do
    nohup python utils/trans20.py $1/raw_${i} $2 ${i} > processing.log 2>&1 &
done

python utils/unify.py $2



