#!/bin/bash
datasets=( "child" "adult")

numones=(  6 7 8 )
#numones=(  5  )

#datasets=("covtype" "credit" "grid" "gridr")
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
  for num in "${numones[@]}"
  do
    cmd="python exp_hd.py --model 2 --dataset ${data} --transformer 0 --numones ${num}"
    echo ${cmd}
    eval ${cmd}
  done
done