#!/bin/bash
set -e
TRIALS=${1:-2}
PROCS=${2:-5}

python3 experiments.py -c ibvs -t $TRIALS -p $PROCS  # creates raw dataset
python3 process_data.py --nprocs $PROCS             # combines dataset into one file
