#!/bin/bash
export PYTHONWARNINGS="ignore"



for run_names in 1697293673 1697296087 1697297687 1697298663 1697299099 1697300189 1697301644 1697303700 1697304939
do
python MLP_PGD.py \
        --run_name $run_names \
        --bs 1000 \
        --epsilons 0 0.5 2.5 5 7.5 10 14 20 30
done