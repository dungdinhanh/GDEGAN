#!/bin/bash
for i in  70
do
#   echo "Latent size: $i"
   for j in 0.0 0.3
   do
#     echo "alpha: ${j}"
    cmd="python main_dae_tuning.py --dataset census --latent ${i} --alpha ${j}"
    echo ${cmd}
    eval ${cmd}
  done

done