#!/bin/bash
export PYTHONWARNINGS="ignore"



for run_names in 1696930897 1696932144 1696936674 1696941636 1696947668 1696954063 1697043800 1697084768 1697091799 1697134502
do
python S2CNN_PGD.py \
        --run_name $run_names \
        --epsilons 0 0.5 2.5 5 7.5 10 14 20 30 \
        --bs 40
done