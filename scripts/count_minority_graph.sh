#!/bin/bash
datasets=("covtype" "adult")
#datasets=("covtype" "credit" "grid" "gridr")
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp_count.py --model 3 --dataset ${data} --transformer 0 --cuda 1"
    echo ${cmd}
    eval ${cmd}
done