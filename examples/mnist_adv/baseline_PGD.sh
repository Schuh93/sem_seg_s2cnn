#!/bin/bash
export PYTHONWARNINGS="ignore"



for run_names in 1671715247 1671716139 1671717283 1671720334 1671724685 1671725735 1671729865 1671742022 1671748092 1671760833
do
python baseline_PGD.py \
        --run_name $run_names \
        --rel_stepsize 0.05 \
        --steps 50
done