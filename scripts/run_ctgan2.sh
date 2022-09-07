#!/bin/bash
datasets=("alarm" "asia" "covtype" "adult")
#datasets=("covtype" "credit" "grid" "gridr")
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp.py --model 0 --dataset ${data} --transformer 0"
    echo ${cmd}
    eval ${cmd}
done