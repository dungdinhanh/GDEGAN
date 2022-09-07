#!/bin/bash
datasets=( "alarm" "insurance")

#numones=( 10 )
numones=( 0 10 )
#numones=(  5  )

#datasets=("covtype" "credit" "grid" "gridr")
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
  for num in "${numones[@]}"
  do
    cmd="python exp_hd.py --model 1 --dataset ${data} --transformer 0 --numones ${num} --iters 1"
    echo ${cmd}
    eval ${cmd}
  done
done