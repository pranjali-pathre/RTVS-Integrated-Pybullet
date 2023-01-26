#!/bin/bash
set -e
TRIALS=2
PROCS=1

python3 experiments.py -c ibvs -t $TRIALS -p $PROCS  # creates raw dataset
python3 load_data.py --nprocs $PROCS             # combines dataset into one file
