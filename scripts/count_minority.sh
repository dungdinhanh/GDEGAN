#!/bin/bash
datasets=( "adult" "covtype" )
#datasets=("covtype" "credit" "grid" "gridr")
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp_count.py --model1 0 --model2 3 --dataset ${data} --transformer 0 "
    echo ${cmd}
    eval ${cmd}
done