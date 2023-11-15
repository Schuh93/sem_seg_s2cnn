#!/bin/bash
export PYTHONWARNINGS="ignore"



for run_names in 1697650878 1697652505 1697658380 1697663547 1697672621 1697673977 1697676458 1697679758
do
python baseline_PGD_mixed_data.py \
        --run_name $run_names \
        --rel_stepsize 0.05 \
        --epsilons 0 0.5 2.5 5 7.5 10 14 20 30 \
        --steps 100
done