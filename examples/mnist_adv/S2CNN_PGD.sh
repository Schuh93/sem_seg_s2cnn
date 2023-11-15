#!/bin/bash
export PYTHONWARNINGS="ignore"



for run_names in 1667922245 1667928002 1667949243 1667987238 1668051244 1668063693 1668085318 1668110081
do
python S2CNN_PGD.py \
        --run_name $run_names \
        --bs 20
done