#!/bin/bash
for i in  70
do
#   echo "Latent size: $i"
   for j in 0.5 0.7
   do
#     echo "alpha: ${j}"
    cmd="python main_dae_tuning.py --dataset census --latent ${i} --alpha ${j}"
    echo ${cmd}
    eval ${cmd}
  done

done