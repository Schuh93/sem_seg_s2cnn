#!/bin/bash
export PYTHONWARNINGS="ignore"


for run_names in 1697263106
do
python DeepFool.py \
        --run_name $run_names \
        --bs 100 \
        --epsilons 0 0.5 2.5 5 7.5 10 14 20 30
done