#!/bin/bash
set -e

python sim.py --no-gui --record
cp imgs/00082.png ./dest.png
