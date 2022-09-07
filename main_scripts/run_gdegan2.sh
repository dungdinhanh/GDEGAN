#!/bin/bash
datasets=( "covtype" "credit" "grid" "gridr" "insurance" "intrusion")
#datasets=("covtype" "credit" "grid" "gridr")
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp.py --model 4 --dataset ${data} --transformer 0 --pac 1"
    echo ${cmd}
    eval ${cmd}
done