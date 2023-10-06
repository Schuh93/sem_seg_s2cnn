#!/bin/bash
export PYTHONWARNINGS="ignore"



for run_names in 1668279489 1668283177 1668284221 1668289478 1668296690 1668302862 1668309303 1668319357 1668343081 1668361949 1667820145 1667706428 1667745011 1667634222 1667729890 1667599289 1668278459
do
python baseline_PGD_mixed_mirrored_data.py \
        --run_name $run_names \
        --rel_stepsize 0.05 \
        --steps 50
done