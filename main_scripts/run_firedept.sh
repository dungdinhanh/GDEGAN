#!/bin/bash
#datasets=( "adult" "census" "covtype" "intrusion" "credit")
#datasets=("covtype" "credit" "grid" "gridr")
models=( 0 8  )
for model in "${models[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp_componly.py --model ${model} --dataset firedept --transformer 2 --pac 10"
    echo ${cmd}
    eval ${cmd}
done