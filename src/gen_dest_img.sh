#!/bin/bash
set -e

python3 sim.py --no-gui --record
cp imgs/00099.png ./dest.png
