#!/bin/bash
export PYTHONWARNINGS="ignore"
       
            
for train_samples in 10000 20000 30000 40000 50000
do
    for in in {1..3}
    do
        python train_S2CNN.py \
                    --name "adv_study" \
                    --channels 8 15 24 20 16 \
                    --bandlimit 30 15 8 4 2 \
                    --kernel_max_beta 0.0625 0.0625 0.125 0.25 0.5 \
                    --nodes 64 32 \
                    --train_samples $train_samples \
                    --activation_fn "LeakyReLU";
    done
done


for train_samples in 10000 20000 30000 40000 50000
do
    for in in {1..3}
    do
        python train_S2CNN.py \
                    --name "adv_study" \
                    --channels 8 12 15 18 128 \
                    --bandlimit 30 15 8 4 2 \
                    --kernel_max_beta 0.0625 0.0625 0.125 0.25 0.5 \
                    --nodes 64 32 \
                    --train_samples $train_samples \
                    --activation_fn "LeakyReLU";
    done
done