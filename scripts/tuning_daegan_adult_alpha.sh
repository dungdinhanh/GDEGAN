#!/bin/bash
for i in 35
do
#   echo "Latent size: $i"
   for j in 0.0 0.1 0.3
   do
#     echo "alpha: ${j}"
    cmd="python main_dae_tuning.py --dataset adult --latent ${i} --alpha ${j}"
    echo ${cmd}
    eval ${cmd}
  done

done