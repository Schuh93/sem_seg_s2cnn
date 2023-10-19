#!/bin/bash
export PYTHONWARNINGS="ignore"



for run_names in 1697522106 1697556854 1697557558 1697562076 1697563004 1697575214 1697581673 1697587785 1697595752 1697602209 1697615182
do
python MLP_PGD.py \
        --run_name $run_names \
        --bs 1000 \
        --epsilons 0 0.5 2.5 5 7.5 10 14 20 30
done