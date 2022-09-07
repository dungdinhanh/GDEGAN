#!/bin/bash
datasets=( "fashion_mnist" "mnist28")
#datasets=("covtype" "credit" "grid" "gridr")
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp.py --model 1 --dataset ${data} --transformer 0 --pac 1"
    echo ${cmd}
    eval ${cmd}
done