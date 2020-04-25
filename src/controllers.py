import os
import time
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
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

# TODO - try Dubin's curves for computing minimum distance
def closest_segment(state, trajectory, N):
    start = np.linalg.norm(trajectory[:, :2] - state[:2], axis=1).argmin()
    end = (start + N) % trajectory.shape[0]
    return start, end

def random_control(u_min, u_max):
    u = (u_max - u_min) * np.random.rand(3) + u_min
    return u

def proportional_control(state, trajectory, u_min, u_max):
    # controller gains
    STEER_GAIN = 0.35
    ACCEL_GAIN = 0.05
    BRAKE_GAIN = 0.05

    # unpack state vector
    x, y, theta, v_x, v_y, omega = state[:6]

    # create waypoint
    start, end = closest_segment(state, trajectory, trajectory.shape[0]//100)
    waypoint = trajectory[end, :2]

    # get waypoint in body frame
    g = twist_to_transform([x, y, theta])
    g_inv = inverse_transform(g)
    point = transform_point(g_inv, waypoint)
    r, psi = polar(point)

    # actual linear velocity
    v = np.linalg.norm([v_x, v_y])

    # desired linear velocity
    v_d = np.linalg.norm([trajectory[start, 3:5]])
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
    u = np.array([steer, accel, min(brake, 0.85)])

    # saturate control
    u = np.clip(u, u_min, u_max)
    return u

# TODO - multiprocessing / linearized model to speed up computations
def model_predictive_control(state, trajectory, u_min, u_max, optimizer, model, dt, H=8, N=4):
    # create reference trajectory
    start, end = closest_segment(state, trajectory, (H+1) * N)

    # handle index wrap
    if start > end:
        trajectory = np.vstack((trajectory[start:], trajectory[:end]))[::N]
    else:
        trajectory = trajectory[start:end:N]

    # set up variables for optimizer
    ts = dt * N
    z_init = state[:6]
    z_ref = trajectory.T

    # run optimizer
    u_opt = optimizer.run(z_init, z_ref, u_min, u_max, H, model, ts)

    #plot_mpc(u_opt.T, z_init, z_ref.T, H, model, ts)

    return u_opt[:, 0]

def plot_mpc(u_opt, z_init, z_ref, H, model, dt, delta=0.1):
    plt.figure(num='mpc', clear=True)
    
    z_opt = np.zeros((H+1, 6))
    z_opt[0] = z_init

    for k in range(H):
        f = model.predict([z_opt[k].reshape(1, -1), u_opt[k].reshape(1, -1)])[0]
        z_opt[k+1, :3] = z_opt[k, :3] + z_opt[k, 3:]*dt
        z_opt[k+1, 2] = z_opt[k+1, 2] % (2*np.pi)
        z_opt[k+1, 3:] = z_opt[k, 3:] + f*dt

    # control
    plt.subplot(2, 1, 1)
    plt.title('Model Predictive Control (H = {})'.format(H))
    plt.plot(range(1, H+1), u_opt[:, 0], '-o', color='C4', markersize=4, label='steering')
    plt.plot(range(1, H+1), u_opt[:, 1], '-o', color='C2', markersize=4, label='throttle')
    plt.plot(range(1, H+1), u_opt[:, 2], '-o', color='C3', markersize=4, label='brake')
    plt.xlabel('step')
    plt.xlim(1 - delta, H + delta)
    plt.ylim(-1 - delta, 1 + delta)
    plt.legend(loc='upper right')
    plt.grid()

    # position
    plt.subplot(2, 2, 3)
    plt.plot(z_ref[:, 0], z_ref[:, 1], '-o', markersize=4, label='ref')
    plt.plot(z_opt[:, 0], z_opt[:, 1], '--o', markersize=4, label='opt')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    plt.grid()

    # orientation
    plt.subplot(2, 2, 4)
    plt.plot(np.arange(1, H+2), z_ref[:, 2], '-o', markersize=4, label='ref')
    plt.plot(np.arange(1, H+2), z_opt[:, 2], '--o', markersize=4, label='opt')
    plt.xlabel('step')
    plt.ylabel('theta')
    plt.xlim(1 - delta, H + delta)
    plt.ylim(0 - delta, 2*np.pi + delta)
    plt.legend(loc='upper right')
    plt.grid()

    plt.pause(1e-3)

class RobotController:
    def __init__(self, model, env, frequency=25):
        self.action = np.array([0.0, 0.0, 0.0])
        self.done = False
        self.model = model
        self.optimizer = NonlinearOptimizer()
        self.trajectory = plan_trajectory(env)
        self.dt = env.dt
        self.z_min = env.observation_space.low
        self.z_max = env.observation_space.high
        self.u_min = env.action_space.low
        self.u_max = env.action_space.high
        self.tick = int(1/(frequency*env.dt))

        env.viewer.window.on_key_press = self.on_key_press

    def on_key_press(self, k, mod):
        if k == key.Q:
            self.done = True

    def step(self, state, t):
        if int(t/self.dt) % self.tick == 0:
            #self.action = random_control(self.u_min, self.u_max)
            #self.action = proportional_control(state, self.trajectory, self.u_min, self.u_max)
            self.action = model_predictive_control(state, self.trajectory,
                                                   self.u_min, self.u_max, 
                                                   self.optimizer, self.model, self.dt)
            
        return self.action, self.done

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
