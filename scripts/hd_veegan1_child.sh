#!/bin/bash
datasets=( "child" )
numones=( 6 7 8 )
#datasets=("covtype" "credit" "grid" "gridr")
for data in "${datasets[@]}"
do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
  for num in "${numones[@]}"
  do
    cmd="python exp_hd.py --model 0 --dataset ${data} --transformer 0 --numones ${num}"
    echo ${cmd}
    eval ${cmd}
  done
done