#!/bin/bash
for i in 35
do
#   echo "Latent size: $i"
   for j in 0.0 0.1 0.3  0.5 0.7 0.9 1.0
   do
#     echo "alpha: ${j}"
    cmd="python main_dae_tuning.py --dataset adult --latent ${i} --alpha ${j} --test"
    echo ${cmd}
    eval ${cmd}
  done

done