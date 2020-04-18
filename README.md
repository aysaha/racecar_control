# Self-Driving Racecar
EECS 206B Final Project

## Download
``git clone https://github.com/aysaha/racecar_control.git --recurse-submodules``

## Installation
``pip install -r requirements.txt -e './gym[box2d]'``

## Learning System Dynamics
``python src/learning.py --dataset data/dynamics.npz  --model models/dynamics.h5` --epochs 100

## Running Simulation
Computer: `python src/main.py`  
User (Keyboard): `python src/main.py --control keyboard`  
User (Xbox Controller): `python src/main.py --control xbox`  

## Development
Main: `src/main.py`  
Learning: `src/learning.py`  
Planning: `src/planning.py`  
Controllers:`src/controllers.py`  
Environment: `gym/gym/envs/box2d/car_racing.py`  
Dynamics: `gym/gym/envs/box2d/car_dynamics.py`

## Reference Equations
control: `u_n = [steering, throttle, brake]`
- -1 <= steering <= 1
- 0 <= throttle <= 1
- 0 <= brake <= 1

state: `z_n = [x, y, theta, v_x, v_y, omega, phi, w_fl, w_fr, w_rl, w_rr] = [q, q_bar]`
- `q_n = [x, y, theta]` - world position
- `q_dot_n = [v_x, v_y, omega]` - world velocity
- `q_bar_n = [q_dot, phi, w_fl, w_fr, w_rl, w_rr]` - velocity with steering angle and individual wheel speeds

system: `z_n+1 = z_n + F(z_n, u_n) * dt`  
kinematics: `q_n+1 = q_n + q_dot_n * dt`  
dynamics: `q_bar_n+1 = q_bar_n + f(z_n, u_n) * dt`  

goal is to learn `f(z_n, u_n) = (q_bar_n+1 - q_bar_n) / dt`
