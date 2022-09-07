#!/bin/bash
datasets=("alarm" "asia" "child" "insurance")

for data in "${datasets[@]}"
do
#   echo "Latent size: $i"

#     echo "alpha: ${j}"
    cmd="python main_ctgan2.py --dataset ${data}"
    echo ${cmd}
    eval ${cmd}


done