#!/bin/bash
set -e

python3 sim.py -c rtvs --no-gui --record --flowdepth
./mkvideo.sh
