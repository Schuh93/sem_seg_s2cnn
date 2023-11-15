#!/bin/bash
export PYTHONWARNINGS="ignore"



for run_names in 1669374570
do
python baseline_PGD.py \
        --run_name $run_names \
        --bs 10
done