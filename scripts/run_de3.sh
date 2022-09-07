#!/bin/bash
datasets=("grid" "gridr" "news" "credit")
#datasets=("gridr" "news" "credit")
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp.py --model 1 --dataset ${data} --transformer 1 --latent 25 --alpha 1.0"
    echo ${cmd}
    eval ${cmd}
done