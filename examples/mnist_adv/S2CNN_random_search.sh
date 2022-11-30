#!/bin/bash
export PYTHONWARNINGS="ignore"

for run_name in 1668559990 1668568059 1668578286 1668598496 1668637573 1668139362 1668030960
do
python S2CNN_random_search.py \
        --run_name $run_name
done