#!/bin/bash
datasets=( "adult" "census" "covtype" "intrusion" "credit")
#datasets=("covtype" "credit" "grid" "gridr")
#models=( 0 2 )
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp.py --model 4 --dataset ${data} --transformer 2 --pac 1 --graph 5"
    echo ${cmd}
    eval ${cmd}
done