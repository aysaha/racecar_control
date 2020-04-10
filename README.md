# racecar_control
EECS 206B Final Project

## Download
``
git clone https://github.com/aysaha/racecar_control.git --recurse-submodules
``

## Installation
``
pip install -r requirements.txt -e './gym[box2d]'
``

## Demo Program
``
python src/demo.py
``

Use the arrow keys to:
- accelerate
- brake
- steer left
- steer right

The sensors at the bottom of the screen measure the following from left to right:
1. front wheel speeds
2. rear wheel speeds
3. chassis speed
4. front wheel angle
5. chassis angular velocity

## Development
Environment: `gym/gym/envs/box2d/car_racing.py`

Dynamics: `gym/gym/envs/box2d/car_dynamics.py`
