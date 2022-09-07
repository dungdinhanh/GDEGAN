#!/bin/bash
for i in 20 25 30
do
#   echo "Latent size: $i"
   for j in 0.1
   do
#     echo "alpha: ${j}"
    cmd="python main_dae_tuning.py --dataset adult --latent ${i} --alpha ${j}"
    echo ${cmd}
    eval ${cmd}
  done

done