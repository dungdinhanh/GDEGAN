#!/bin/bash
datasets=("child" "insurance" "alarm" "asia" )
#datasets=("covtype" "credit" "grid" "gridr")
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp.py --model 3 --dataset ${data} --transformer 0 --graph 1"
    echo ${cmd}
    eval ${cmd}
done