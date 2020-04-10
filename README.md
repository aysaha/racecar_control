# racecar_control
EECS 206B Final Project

## Installation
Environment:
```
cd gym
pip install -e '.[box2d]'
```

Dependencies:
```
pip install -r requirements.txt
```


## Demo Program
Run:
```
python src/demo.py
```

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
