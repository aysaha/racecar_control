import os
import time
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from pyglet.window import key
import inputs
import casadi

from utils import *

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

def initialize_controller(control, trajectory, env):
    print("[{}] initializing controller ({})".format(FILE, control))

    if control == 'robot':
        controller = RobotController(trajectory, env)
    elif control == 'keyboard':
        controller = KeyboardController(env)
    elif control == 'xbox':
        controller = XboxController()
    else:
        controller = None

    return controller

class RobotController:
    def __init__(self, trajectory, env):
        self.action = np.array([0.0, 0.0, 0.0])
        self.done = False
        self.trajectory = trajectory
        self.dt = env.dt
        self.z_min = env.observation_space.low
        self.z_max = env.observation_space.high
        self.u_min = env.action_space.low
        self.u_max = env.action_space.high
        self.z = None
        self.u = None
        self.random = False

        env.viewer.window.on_key_press = self.on_key_press

    def on_key_press(self, k, mod):
        if k == key.Q:
            self.done = True

    def random_control(self):
        u = (self.u_max - self.u_min) * np.random.rand(3) + self.u_min
        return u

    def proportional_control(self, state):
        # controller gains
        STEER_GAIN = 0.35
        ACCEL_GAIN = 0.05
        BRAKE_GAIN = 0.05

        # unpack state vector
        x, y, theta, v_x, v_y, omega = state[:6]

        # waypoint horizon
        N = self.trajectory.shape[0]
        H = N // 100

        # create waypoint
        start = np.linalg.norm(self.trajectory[:, :2] - [x, y], axis=1).argmin()
        end = (start + H) % N
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
        u = np.array([steer, accel, min(brake, 0.85)])

        # saturate control
        u = np.clip(u, self.u_min, self.u_max)
        return u

    def model_predictive_control(self, state, t, H=50):
        def plot_mpc(q_ref, q_opt, u_opt, H):
            assert q_ref.shape == (H+1, 3)
            assert q_opt.shape == (H+1, 3)
            assert u_opt.shape == (H, 3)
            plt.figure(num='mpc')

            # control
            plt.subplot(2, 1, 1)
            plt.title('Model Predictive Control (H = {})'.format(H))
            plt.plot(np.arange(H), u_opt[:, 0], '-o', color='C4', markersize=4, label='steering')
            plt.plot(np.arange(H), u_opt[:, 1], '-o', color='C2', markersize=4, label='throttle')
            plt.plot(np.arange(H), u_opt[:, 2], '-o', color='C3', markersize=4, label='brake')
            plt.xlabel('step')
            plt.ylim(-1, 1)
            plt.legend()
            plt.grid()

            # position
            plt.subplot(2, 2, 3)
            plt.plot(q_ref[:, 0], q_ref[:, 1], '-o', markersize=4, label='ref')
            plt.plot(q_opt[:, 0], q_opt[:, 1], '--o', markersize=4, label='opt')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.grid()

            # orientation
            plt.subplot(2, 2, 4)
            plt.plot(np.arange(H+1), q_ref[:, 2], '-o', markersize=4, label='ref')
            plt.plot(np.arange(H+1), q_opt[:, 2], '--o', markersize=4, label='opt')
            plt.xlabel('step')
            plt.ylabel('theta')
            plt.ylim(0, 2*np.pi)
            plt.legend()
            plt.grid()

            plt.show()

        def dynamics(z, u):
            assert z.shape == (11, 1)
            assert u.shape == (3, 1)

            tensor = casadi.vertcat(z, u)
            layers = self.agent.model.get_weights()
            L = len(layers)

            for i in range(0, L, 2):
                if i < L-2:
                    tensor = casadi.tanh(layers[i].T @ tensor  + layers[i+1])
                else:
                    tensor = layers[i].T @ tensor  + layers[i+1]

            return tensor

        def J(z, u, q_ref, H):
            assert z.shape == (11, H+1)
            assert u.shape == (3, H)
            assert q_ref.shape == (3, H+1)

            Q = np.diag([10, 10, 1])
            R = np.diag([1, 1, 1])
            cost = 0

            for k in range(H):
                cost += (z[:3, k] - q_ref[:, k]).T @ Q @ (z[:3, k] - q_ref[:, k]) + u[:, k].T @ R @ u[:, k]

                f = dynamics(z[:, k], u[:, k])
                z[:3, k+1] = z[:3, k] + z[3:6, k]*self.dt
                z[2, k+1] = casadi.fmod(z[2, k+1], 2*np.pi)
                z[3:, k+1] = z[3:, k] + f*self.dt

            cost += (z[:3, H] - q_ref[:, H]).T @ Q @ (z[:3, H] - q_ref[:, H])

            return cost

        N = int(t/self.dt)

        if N % H == 0 or self.u is None:
            print("[{}] running optimizer (t = {}s)".format(FILE, int(t)))

            # create reference trajectory
            index = np.linalg.norm(self.trajectory[:, :2] - state[:2], axis=1).argmin()
            start = index
            end = (start + (H+1)) % self.trajectory.shape[0]

            if start > end:
                q_ref = np.vstack((self.trajectory[start:], self.trajectory[:end])).T
            else:
                q_ref = self.trajectory[start:end].T

            z_init = state
            u_init = np.zeros((3, H))
            u_init[1, :] = 0.5

            opti = casadi.Opti()

            z = opti.variable(11, H+1)
            opti.set_initial(z[:3, :], q_ref)

            u = opti.variable(3, H)
            opti.set_initial(u, u_init)
            
            opti.minimize(J(z, u, q_ref, H))
            opti.subject_to(z[:, 0] == z_init)

            low = np.tile(self.u_min, H)
            high = np.tile(self.u_max, H)
            bounds = opti.bounded(low, casadi.vec(u), high)
            opti.subject_to(bounds)
            
            opti.solver('ipopt', {'print_time': False}, {'print_level': 0})
            sol = opti.solve()
            z_opt = sol.value(z)
            u_opt = sol.value(u)

            self.z = z_opt.T
            self.u = u_opt.T

            #plot_mpc(q_ref.T, self.z[:3], self.u, H)

        return self.u[N % H]

    def step(self, state, t):
        if int(t/self.dt) % int(1/self.dt) == 0:
            self.random = np.random.rand() < 0.5
            self.action = self.random_control()

        if not self.random:
          self.action = self.proportional_control(state)

        #self.action = self.model_predictive_control(state, t)

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
