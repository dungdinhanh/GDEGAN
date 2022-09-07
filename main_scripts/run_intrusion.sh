#!/bin/bash
datasets=( "intrusion" )
#datasets=("covtype" "credit" "grid" "gridr")
models=( 6 3 )
for data in "${datasets[@]}"
do
  for model in "${models[@]}"
  do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp_componly.py --model ${model} --dataset ${data} --transformer 0 --iters 0"
    echo ${cmd}
    eval ${cmd}
    done
done