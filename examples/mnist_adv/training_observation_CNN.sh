#!/bin/bash
export PYTHONWARNINGS="ignore"

python training_observation_CNN.py \
        --name "adv_study" \
        --channels 16 24 32 64 \
        --kernels 3 3 3 3 \
        --strides 1 1 1 1 \
        --nodes 64 32 \
        --train_samples 10000 \
        --activation_fn "LeakyReLU"
