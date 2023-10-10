#!/bin/bash
export PYTHONWARNINGS="ignore"



for run_names in 1668559990 1668568059 1668578286 1668598496 1668637573 1668139362 1668030960
do
python S2CNN_PGD_mixed_mirrored_data.py \
        --run_name $run_names
done