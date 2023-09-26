#!/bin/bash
export PYTHONWARNINGS="ignore"



for run_names in 1686568319 1686568791 1686572608 1686575304 1686579498 1686584727 1686591061 1686618156 1686625770 1686649185
do
python baseline_PGD.py \
        --run_name $run_names \
        --rel_stepsize 0.05 \
        --steps 50
done