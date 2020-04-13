# Self-Driving Racecar
EECS 206B Final Project

## Download
``git clone https://github.com/aysaha/racecar_control.git --recurse-submodules``

## Installation
``pip install -r requirements.txt -e './gym[box2d]'``

## Running
#### Playing with a Keyboard
``python src/main.py -c keyboard``

#### Playing with an Xbox Controller
``python src/main.py -c xbox``

#### Learning System Dynamics
``python src/learning.py -d data/dynamics.npz  -m models/dynamics.h5``

## Development
Main: `src/main.py`  
Controllers:`src/controllers.py`  
Learning: `src/learning.py`  
Environment: `gym/gym/envs/box2d/car_racing.py`  
Dynamics: `gym/gym/envs/box2d/car_dynamics.py`
