#!/bin/bash
datasets=( "insurance" "intrusion" )
#datasets=("covtype" "credit" "grid" "gridr")
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp.py --model 5 --dataset ${data} --transformer 0"
    echo ${cmd}
    eval ${cmd}
done