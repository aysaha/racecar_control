import os
import time
import multiprocessing

import numpy as np
from pyglet.window import key
import inputs

from optimization import NonlinearOptimizer
from planning import plan_trajectory
from utils import *

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

def initialize_controller(control, model, env):
    print("[{}] initializing controller ({})".format(FILE, control))

    if control == 'robot':
        controller = RobotController(model, env)
    elif control == 'keyboard':
        controller = KeyboardController(env)
    elif control == 'xbox':
        controller = XboxController()
    else:
        controller = None

    return controller

class RobotController:
    def __init__(self, model, env, frequency=10, T=20):
        self.action = np.array([0.0, 0.0, 0.0])
        self.done = False
        self.dt = env.dt
        self.z_min = env.observation_space.low
        self.z_max = env.observation_space.high
        self.u_min = env.action_space.low
        self.u_max = env.action_space.high
        self.frequency = frequency
        self.T = T
        self.tick = int(1/(frequency*env.dt))
        self.optimizer = NonlinearOptimizer(model, ts=1/frequency)
        self.trajectory = plan_trajectory(env, T=T)
        
        env.viewer.window.on_key_press = self.on_key_press

        # default linear solver: 'mumps'
        # external linear solvers: 'ma27', 'ma57', 'ma77', 'ma86', 'ma97'
        # available from http://www.hsl.rl.ac.uk/ipopt/
        self.optimizer.options['linear_solver'] = 'ma57'

    def on_key_press(self, k, mod):
        if k == key.Q:
            self.done = True

    def update_trajectory(self, env, T):
        self.T = T
        self.trajectory = plan_trajectory(env, T=T)

    def closest_segment(self, state, N):
        start = np.linalg.norm(self.trajectory[:, :2] - state[:2], axis=1).argmin()
        end = (start + N) % self.trajectory.shape[0]
        return start, end

    def step(self, state, t):
        if int(t/self.dt) % self.tick == 0:
            #start = time.time()
            self.action = self.model_predictive_control(state)
            #self.action = self.proportional_control(state)
            #self.action = self.random_control()
            #end = time.time()
            
            #print("[{}] t_step = {:.3f}s".format(FILE, end - start))
            
        return self.action, self.done

    def model_predictive_control(self, state, H=5):
        # controller gains
        K = np.diag([0.05, 0.01, 0.01])
        P = np.diag([100, 100, 1, 1, 1, 1])
        Q = np.diag([100, 100, 1, 1, 1, 1])
        R = np.diag([1, 1, 1])

        # create reference trajectory
        start, end = self.closest_segment(state, (H+1)*self.tick)
        
        # handle index wrap
        if start > end:
            trajectory = np.vstack((self.trajectory[start:], self.trajectory[:end]))
        else:
            trajectory = self.trajectory[start:end]

        # create waypoint
        waypoint = self.trajectory[(start + self.tick) % self.trajectory.shape[0], :2]

        # transform waypoint to car frame
        g = twist_to_transform(state[:3])
        point = transform_point(inverse_transform(g), waypoint)
        r, psi = polar(point)

        # determine velocity error
        v = np.linalg.norm(state[3:5])
        v_d = np.linalg.norm([self.trajectory[start, 3:5]]) * abs(np.cos(psi))
        e = v_d - v

        # compute control
        u = np.clip(K @ [-psi, e, -e], self.u_min, self.u_max)

        # run optimizer
        z_init = state[:6]
        z_ref = trajectory[::self.tick].T
        u_init = np.tile(u, (H, 1)).T
        u_opt = self.optimizer.run(z_init, z_ref, u_init, self.u_min, self.u_max, P, Q, R, H)

        # prevent stalling
        if v < 10 and u_opt[1, 0] - u_opt[2, 0] < 0.1:
            u_opt[:, 0] = [1.0*np.sign(u[0]), 0.1, 0.0]

        return u_opt[:, 0]

    def proportional_control(self, state):
        # controller gains
        STEER_GAIN = 0.35
        ACCEL_GAIN = 0.05
        BRAKE_GAIN = 0.05

        # unpack state vector
        x, y, theta, v_x, v_y, omega = state[:6]

        # create waypoint
        start, end = self.closest_segment(state, self.trajectory.shape[0]//100)
        waypoint = self.trajectory[end, :2]

        # get waypoint in body frame
        g = twist_to_transform([x, y, theta])
        g_inv = inverse_transform(g)
        point = transform_point(g_inv, waypoint)
        r, psi = polar(point)

        # actual linear velocity
        v = np.linalg.norm([v_x, v_y])

        # desired linear velocity
        v_d = np.linalg.norm([self.trajectory[start, 3:5]])
        v_d *= abs(np.cos(psi))

        # control is proportional to the error
        steer = STEER_GAIN * -psi
        accel = ACCEL_GAIN * (v_d - v)
        brake = BRAKE_GAIN * (v - v_d)

        # prevent stalling
        if v < 10:
            accel = 0.1
            brake = 0.0

        # limit brakes to prevent locking wheels
        u = np.array([steer, accel, min(brake, 0.8)])

        # saturate control
        u = np.clip(u, self.u_min, self.u_max)
        return u

    def random_control(self):
        u = (self.u_max - self.u_min) * np.random.rand(3) + self.u_min
        return u

class KeyboardController:
    LEFT = -1.0
    RIGHT = 1.0
    ACCELERATE = 1.0
    BRAKE = 0.8

    def __init__(self, env):
        self.action = np.array([0.0, 0.0, 0.0])
        self.done = False

        env.viewer.window.on_key_press = self.on_key_press
        env.viewer.window.on_key_release = self.on_key_release

    def on_key_press(self, k, mod):
        if k == key.LEFT: self.action[0] = KeyboardController.LEFT
        if k == key.RIGHT: self.action[0] = KeyboardController.RIGHT
        if k == key.UP: self.action[1] = KeyboardController.ACCELERATE
        if k == key.DOWN: self.action[2] = KeyboardController.BRAKE
        if k == key.Q: self.done = True

    def on_key_release(self, k, mod):
        if k == key.LEFT and self.action[0] == KeyboardController.LEFT: self.action[0] = 0.0
        if k == key.RIGHT and self.action[0] == KeyboardController.RIGHT: self.action[0] = 0.0
        if k == key.UP: self.action[1] = 0.0
        if k == key.DOWN: self.action[2] = 0.0

    def step(self, state, t):
        return self.action, self.done

class XboxController:
    MAX_JOY_VAL = np.power(2, 15) - 1
    MAX_TRIG_VAL = np.power(2, 8) - 1

    def __init__(self):
        assert len(inputs.devices.gamepads) == 1

        self.LeftJoystickX = multiprocessing.Value('i', 0)
        self.LeftJoystickY = multiprocessing.Value('i', 0)
        self.RightJoystickX = multiprocessing.Value('i', 0)
        self.RightJoystickY = multiprocessing.Value('i', 0)
        self.LeftTrigger = multiprocessing.Value('i', 0)
        self.RightTrigger = multiprocessing.Value('i', 0)
        self.LeftBumper = multiprocessing.Value('i', 0)
        self.RightBumper = multiprocessing.Value('i', 0)
        self.A = multiprocessing.Value('i', 0)
        self.B = multiprocessing.Value('i', 0)
        self.X = multiprocessing.Value('i', 0)
        self.Y = multiprocessing.Value('i', 0)
        self.LeftThumb = multiprocessing.Value('i', 0)
        self.RightThumb = multiprocessing.Value('i', 0)
        self.Back = multiprocessing.Value('i', 0)
        self.Start = multiprocessing.Value('i', 0)
        self.LeftDPad = multiprocessing.Value('i', 0)
        self.RightDPad = multiprocessing.Value('i', 0)
        self.UpDPad = multiprocessing.Value('i', 0)
        self.DownDPad = multiprocessing.Value('i', 0)
        self.process = multiprocessing.Process(target=self.monitor_controller)

        print("[{}] starting daemon process".format(FILE))
        self.process.daemon = True
        self.process.start()
        time.sleep(3)

    def monitor_controller(self):
        print("[{}] daemon process started".format(FILE))
        while True:
            time.sleep(0.001)
            events = inputs.get_gamepad()

            for event in events:
                if event.code == 'ABS_X': self.LeftJoystickX.value = event.state
                elif event.code == 'ABS_Y': self.LeftJoystickY.value = event.state
                elif event.code == 'ABS_RX': self.RightJoystickX.value = event.state
                elif event.code == 'ABS_RY': self.RightJoystickY.value = event.state
                elif event.code == 'ABS_Z': self.LeftTrigger.value = event.state
                elif event.code == 'ABS_RZ': self.RightTrigger.value = event.state
                elif event.code == 'BTN_TL': self.LeftBumper.value = event.state
                elif event.code == 'BTN_TR': self.RightBumper.value = event.state
                elif event.code == 'BTN_SOUTH': self.A.value = event.state
                elif event.code == 'BTN_EAST': self.B.value = event.state
                elif event.code == 'BTN_NORTH': self.X.value = event.state
                elif event.code == 'BTN_WEST': self.Y.value = event.state
                elif event.code == 'BTN_THUMBL': self.LeftThumb.value = event.state
                elif event.code == 'BTN_THUMBR': self.RightThumb.value = event.state
                elif event.code == 'BTN_SELECT': self.Back.value = event.state
                elif event.code == 'BTN_START': self.Start.value = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY1': self.LeftDPad.value = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY2': self.RightDPad.value = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY3': self.UpDPad.value = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY4': self.DownDPad.value = event.state

    def step(self, state, t):
        steer = np.clip(np.power(self.LeftJoystickX.value / XboxController.MAX_JOY_VAL, 3), -1, 1)
        accel = np.clip(np.power(self.RightTrigger.value / XboxController.MAX_TRIG_VAL, 2), 0, 1)
        brake = np.clip(np.power(self.LeftTrigger.value / XboxController.MAX_TRIG_VAL, 2), 0, 1)
        done = True if self.B.value == 1 else False
        return np.array([steer, accel, brake]), done
