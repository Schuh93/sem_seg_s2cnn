#!/bin/bash
export PYTHONWARNINGS="ignore"

python create_untrained_CNN.py \
        --name "adv_study" \
        --channels 16 24 32 64 \
        --kernels 3 3 3 3 \
        --strides 1 1 1 1 \
        --nodes 64 32 \
        --activation_fn "LeakyReLU"