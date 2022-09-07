#!/bin/bash
datasets=( "intrusion" )
#datasets=("covtype" "credit" "grid" "gridr")
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp_componly.py --model 1 --dataset ${data} --transformer 1"
    echo ${cmd}
    eval ${cmd}
done