#!/bin/bash
export PYTHONWARNINGS="ignore"



for run_names in 1697632034 1697633409 1697635738 1697638647 1697643509 1697644887 1697649945 1697665587 1697670845 1697684622
do
python baseline_PGD_mixed_data.py \
        --run_name $run_names \
        --rel_stepsize 0.05 \
        --epsilons 0 0.5 2.5 5 7.5 10 14 20 30 \
        --steps 100
done