#!/bin/bash
datasets=( "fashion_mnist" )
#datasets=("covtype" "credit" "grid" "gridr")
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp_componly.py --model 7 --dataset ${data} --transformer 0 --iters 0"
    echo ${cmd}
    eval ${cmd}
done