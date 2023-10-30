#!/bin/bash
export PYTHONWARNINGS="ignore"



for run_names in 1697665587 1697670845 1697684622
do
python baseline_PGD.py \
        --run_name $run_names \
        --rel_stepsize 0.05 \
        --steps 50 \
        --epsilons 0 0.5 2.5 5 7.5 10 14 20 30 \
        --bs 100
done