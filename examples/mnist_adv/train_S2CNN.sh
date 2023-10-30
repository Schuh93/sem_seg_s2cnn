#!/bin/bash
export PYTHONWARNINGS="ignore"

            
for train_samples in 30000 40000 50000 60000
do
    for i in {1..3}
    do
        python train_S2CNN.py \
                    --name "adv_study" \
                    --channels 8 15 24 20 16 \
                    --bandlimit 14 11 8 4 2 \
                    --kernel_max_beta  0.07 0.09 0.125 0.25 0.5 \
                    --nodes 64 32 \
                    --train_samples $train_samples \
                    --activation_fn "LeakyReLU" \
                    --padded_img_size 28 28 \
                    --flat;
    done
done


for train_samples in 10000 20000 30000 40000 50000 60000
do
    for i in {1..3}
    do
        python train_S2CNN.py \
                    --name "adv_study" \
                    --channels 8 12 15 18 128 \
                    --bandlimit 14 11 8 4 2 \
                    --kernel_max_beta  0.07 0.09 0.125 0.25 0.5 \
                    --nodes 64 32 \
                    --train_samples $train_samples \
                    --activation_fn "LeakyReLU" \
                    --padded_img_size 28 28 \
                    --flat;
    done
done


for train_samples in 80000 100000 120000 240000 600000
do
    for i in {1..3}
    do
        python train_S2CNN.py \
                    --name "adv_study" \
                    --channels 8 11 64 \
                    --bandlimit 14 8 2 \
                    --kernel_max_beta  0.07 0.125 0.5 \
                    --nodes 64 32 \
                    --train_samples $train_samples \
                    --activation_fn "LeakyReLU" \
                    --padded_img_size 28 28 \
                    --flat;
    done
done


for train_samples in 80000 100000 120000 240000 600000
do
    for i in {1..3}
    do
        python train_S2CNN.py \
                    --name "adv_study" \
                    --channels 8 15 24 20 16 \
                    --bandlimit 14 11 8 4 2 \
                    --kernel_max_beta  0.07 0.09 0.125 0.25 0.5 \
                    --nodes 64 32 \
                    --train_samples $train_samples \
                    --activation_fn "LeakyReLU" \
                    --padded_img_size 28 28 \
                    --flat;
    done
done


for train_samples in 80000 100000 120000 240000 600000
do
    for i in {1..3}
    do
        python train_S2CNN.py \
                    --name "adv_study" \
                    --channels 8 12 15 18 128 \
                    --bandlimit 14 11 8 4 2 \
                    --kernel_max_beta  0.07 0.09 0.125 0.25 0.5 \
                    --nodes 64 32 \
                    --train_samples $train_samples \
                    --activation_fn "LeakyReLU" \
                    --padded_img_size 28 28 \
                    --flat;
    done
done