#!/bin/bash
export PYTHONWARNINGS="ignore"



for run_names in 1697050764 1697052959 1697061474 1697073402 1697096372 1697103093 1697110031 1697121241
do
python S2CNN_PGD.py \
        --run_name $run_names \
        --epsilons 0 0.5 2.5 5 7.5 10 14 20 30 \
        --bs 10
done