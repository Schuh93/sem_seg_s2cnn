#!/bin/bash
export PYTHONWARNINGS="ignore"
       
            
for train_samples in 100000
do
    for i in {1..3}
    do
        python train_S2CNN.py \
                    --name "adv_study" \
                    --channels 8 11 64 \
                    --bandlimit 30 8 2 \
                    --kernel_max_beta 0.0625 0.33 0.5 \
                    --nodes 64 32 \
                    --train_samples $train_samples \
                    --activation_fn "LeakyReLU" \
                    --flat;
    done
done