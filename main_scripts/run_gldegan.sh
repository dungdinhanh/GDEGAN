#!/bin/bash
datasets=( "firedept" "colorado")
#datasets=("covtype" "credit" "grid" "gridr")
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp_componly.py --model 10 --dataset ${data} --transformer 2 --pac 1 --latent 128"
    echo ${cmd}
    eval ${cmd}
done