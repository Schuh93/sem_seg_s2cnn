#!/bin/bash
export PYTHONWARNINGS="ignore"



for run_names in 1668139362 1668030960
do
python S2CNN_PGD.py \
        --run_name $run_names
done