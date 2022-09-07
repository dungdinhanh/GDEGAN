#!/bin/bash
datasets=( "mnist28")
#datasets=("covtype" "credit" "grid" "gridr")
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp_componly.py --model 8 --dataset ${data} --latent 128 --alpha 1.0"
    echo ${cmd}
    eval ${cmd}
done