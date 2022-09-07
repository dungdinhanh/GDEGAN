#!/bin/bash
#datasets=("alarm" "asia" "child" "insurance" "adult")
#datasets=("grid" "gridr" "news" "credit" )
datasets=("grid" "gridr"  )
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    cmd="python exp_daegan_gr.py --dataset ${data} --transformer 0 --latent 25 --alpha 1.0"
    echo ${cmd}
    eval ${cmd}
done