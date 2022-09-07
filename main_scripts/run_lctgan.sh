#!/bin/bash
datasets=( "mnist12"  "mnist28" "fashion_mnist")
#datasets=("covtype" "credit" "grid" "gridr")
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp_componly.py --model 9 --dataset ${data} --transformer 0 --pac 10 "
    echo ${cmd}
    eval ${cmd}
done