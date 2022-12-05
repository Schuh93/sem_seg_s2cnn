#!/bin/bash
export PYTHONWARNINGS="ignore"

for run_name in 1668361949
do
python baseline_random_search.py \
        --run_name $run_name \
        --mode 'max'
done

for run_name in 1668559990
do
python S2CNN_random_search.py \
        --run_name $run_name \
        --mode 'max'
done