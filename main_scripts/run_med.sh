#!/bin/bash
datasets=( "adult" "alarm" "asia" "child" "census" "news" "ring")
#datasets=("covtype" "credit" "grid" "gridr")
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp.py --model 6 --dataset ${data} --transformer 0"
    echo ${cmd}
    eval ${cmd}
done