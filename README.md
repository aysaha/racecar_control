# racecar_control
EECS 206B Final Project

## Download
``git clone https://github.com/aysaha/racecar_control.git --recurse-submodules``

## Installation
``pip install -r requirements.txt -e './gym[box2d]'``

## Running
#### Playing with a Keyboard
``python src/main.py keyboard``

#### Playing with an Xbox Controller
``python src/main.py xbox``

#### Recording System Dynamics
``python src/main.py keyboard -o data/dynamics.npz``

#### Learning System Dynamics
``python src/main.py keyboard data/dynamics.npz  -o models/dynamics.h5``

#### Testing System Dynamics
``python src/main.py keyboard -i models/dynamics.h5``

## Development
Main: `src/learning.py`  
Controllers:`src/controllers.py`  
Learning: `src/learning.py`  
Environment: `gym/gym/envs/box2d/car_racing.py`  
Dynamics: `gym/gym/envs/box2d/car_dynamics.py`
