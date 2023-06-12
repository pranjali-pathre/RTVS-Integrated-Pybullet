# Grasping Moving Object using Visual Servoing

## Setup Environment

- make sure to pull submodules using `git submodule update --init --recursive`.
- conda create env from .yml file
- install airobot using: `pip install git+https://github.com/Improbable-AI/airobot@4c0fe31`
- install flownet2 submodule present in the subdir `src/controllers/rtvs/flownet2` and using `install.sh`

## Basic Explanation

- `src/sim.py` contains the main code intracting with the simulation env and the arm.
- `src/rtvs_run.sh` provides an example how to run the simulation using rtvs controller, and additionaly stiches frames together to a video.
- `src/create_dataset.py` and `src/train.py` are for ml training the depth net, while the main ML code is in `src/depthnet`.
- `src/controllers` are various controllers available.
