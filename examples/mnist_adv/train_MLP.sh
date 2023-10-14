#!/bin/bash
export PYTHONWARNINGS="ignore"


for train_samples in 10000 20000 30000 40000 50000 60000
do
    for i in {1..3}
    do
        python train_MLP.py \
                --name "adv_study" \
                --nodes 44 23 \
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
        python train_MLP.py \
                --name "adv_study" \
                --nodes 60 30 20 \
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
        python train_MLP.py \
                --name "adv_study" \
                --nodes 120 90 64 32 \
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
        python train_MLP.py \
                --name "adv_study" \
                --nodes 300 100 \
                --train_samples $train_samples \
                --activation_fn "LeakyReLU" \
                --padded_img_size 28 28 \
                --batch_norm \
                --flat;
    done
done


for train_samples in 10000 20000 30000 40000 50000 60000
do
    for i in {1..3}
    do
        python train_MLP.py \
                --name "adv_study" \
                --nodes 300 \
                --train_samples $train_samples \
                --activation_fn "LeakyReLU" \
                --padded_img_size 28 28 \
                --batch_norm \
                --flat;
    done
done