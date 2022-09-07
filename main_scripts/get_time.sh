#!/bin/bash
datasets=( "asia" "adult" "census" "colorado" )
#datasets=("covtype" "credit" "grid" "gridr")
models=( 0 6 7 )
for model in "${models[@]}"
do
  for data in "${datasets[@]}"
  do
#   echo "Latent size: $i"
#     echo "alpha: ${j}"
    if [[ $model -eq 4 ]]
    then
      pac=1
    else
      pac=10
    fi
    cmd="python exp_comptimeonly.py --model ${model} --dataset ${data} --transformer 2 --pac ${pac} --outputdir outputtime"
    echo ${cmd}
    eval ${cmd}
  done
done