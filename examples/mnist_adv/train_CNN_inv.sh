#!/bin/bash
export PYTHONWARNINGS="ignore"


for train_samples in 10000 20000 30000 40000 50000 60000 80000 100000 120000 240000 600000
do
    for i in {1..3}
    do
        python train_CNN_inv.py \
                --name "adv_study" \
                --channels 16 24 32 64 \
                --kernels 3 3 3 3 \
                --strides 1 1 1 1 \
                --nodes 64 32 \
                --train_samples $train_samples \
                --activation_fn "LeakyReLU";
    done
done


for train_samples in 10000 20000 30000 40000 50000 60000 80000 100000 120000 240000 600000
do
    for i in {1..3}
    do
        python train_CNN_inv.py \
                --name "adv_study" \
                --channels 16 24 32 64 128 \
                --kernels 3 3 3 3 3 \
                --strides 1 1 1 1 1 \
                --nodes 64 32 \
                --train_samples $train_samples \
                --activation_fn "LeakyReLU";
    done
done


for train_samples in 10000 20000 30000 40000 50000 60000 80000 100000 120000 240000 600000
do
    for i in {1..3}
    do
        python train_CNN_inv.py \
                --name "adv_study" \
                --channels 16 32 64 32 16 \
                --kernels 3 3 3 3 3 \
                --strides 1 1 1 1 1 \
                --nodes 64 32 \
                --train_samples $train_samples \
                --activation_fn "LeakyReLU";
    done
done