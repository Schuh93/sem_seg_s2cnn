#!/bin/bash
export PYTHONWARNINGS="ignore"

for run_name in 1697206030 1697206644 1697207683 1697209477 1697210525 1697213123 1697215830 1697224525 1697226035 1697236373
do
python baseline_random_search.py \
        --run_name $run_name \
        --mode 'max'
done

