#!/bin/bash
set -e

python3 sim.py -c ours --no-gui --record --flowdepth
./mkvideo.sh
