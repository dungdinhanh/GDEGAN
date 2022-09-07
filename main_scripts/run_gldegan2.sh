#!/bin/bash
datasets=(  "colorado")
alphas=("0.3" "0.5")
#datasets=("covtype" "credit" "grid" "gridr")
for data in "${datasets[@]}"
do
  for alpha in "${alphas[@]}"
  do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp_componly.py --model 10 --dataset ${data} --transformer 2 --pac 1 --latent 128 --alpha ${alpha}"
    echo ${cmd}
    eval ${cmd}
  done
done