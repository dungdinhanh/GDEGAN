#!/bin/bash
datasets=( "adult" "census" "covtype" "intrusion" "credit")
#datasets=("covtype" "credit" "grid" "gridr")
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp_stats_dataset.py --model 7 --dataset ${data} --transformer 0 --pac 1"
    echo ${cmd}
    eval ${cmd}
done