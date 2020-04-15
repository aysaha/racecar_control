import os
import time
import multiprocessing

import numpy as np
from scipy import optimize
from pyglet.window import key
import inputs

from learning import dynamics
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
    def __init__(self, model, env):
        self.action = np.array([0.0, 0.0, 0.0])
        self.done = False
        self.model = model
        self.track = np.array(env.track)
        self.q_min = np.array(env.observation_space.low)
        self.q_max = np.array(env.observation_space.high)
        self.u_min = np.array(env.action_space.low)
        self.u_max = np.array(env.action_space.high)
        self.u = None

        env.viewer.window.on_key_press = self.on_key_press

    def on_key_press(self, k, mod):
        if k == key.Q:
            self.done = True

    def random_control(self, state, t):
        if t % 100 == 0:
            u = (self.u_max - self.u_min) * np.random.rand(3) + self.u_min
        else:
            u = self.action

        return u

    def basic_control(self, state, t):
        ANGLE_LIMIT = np.pi/12
        VEL_LIMIT = 1
        STEER_GAIN = 0.25
        ACCEL_GAIN = 0.01
        BRAKE_GAIN = 0.35

        # calculate linear velocity
        vel = np.linalg.norm(state[3:5])

        # create target point
        index = np.linalg.norm(self.track[:, -2:] - state[:2], axis=1).argmin()
        index = (index + 10) % len(self.track)
        waypoint = self.track[index, -2:]

        # get target point in car frame
        g = twist_to_transform(state[:3])
        g_inv = inv(g)
        point = transform_point(g_inv, waypoint)
        d, phi = polar(point)

        if abs(phi) < ANGLE_LIMIT:
            # DRIVE
            steer = STEER_GAIN * -phi
            accel = ACCEL_GAIN * d*np.cos(phi)
            brake = 0.0
        elif abs(phi) < 2*ANGLE_LIMIT:
            # COAST
            steer = 2*STEER_GAIN * -phi
            accel = 0.0
            brake = 0.0
        elif abs(vel) < VEL_LIMIT:
            # SLOW
            steer = 4*STEER_GAIN * -phi
            accel = ACCEL_GAIN/4 * d*np.cos(phi)
            brake =  0.0
        else:
            # BRAKE
            steer = 4*STEER_GAIN * -phi
            accel = 0.0
            brake =  BRAKE_GAIN - 1/np.square(vel)

        # saturate control
        u = np.array([steer, accel, np.clip(brake, 0, 0.8)])
        u = np.clip(u, self.u_min, self.u_max)
        return u

    def model_predictive_control(self, state, t, T=1, H=4):
        def J(u, q_i, q_d, H):
            Q = np.diag([1, 1, 1, 1, 1, 0])
            R = np.diag([1, 1, 1])

            q = q_i
            u = u.reshape((-1, 3))
            cost = 0

            for k in range(H):
                cost += (q - q_d[k]).T @ Q @ (q - q_d[k]) + u[k].T @ R @ u[k]
                q = dynamics(q, u[k], self.model)

            cost += (q - q_d[H]).T @ Q @ (q - q_d[H])

            return cost

        if t % (T*H) == 0:
            #print("[{}] starting optimization".format(FILE))
            
            g = inv(twist_to_transform(state[:3]))

            # create reference trajectory
            index = np.linalg.norm(self.track[:, -2:] - state[:2], axis=1).argmin()
            start = index
            end = (start + (H + 1)) % len(self.track)

            if start > end:
                trajectory = np.vstack((self.track[start:, -2:], self.track[:end, -2:]))
            else:
                trajectory = self.track[start:end, -2:]

            trajectory = np.hstack((trajectory,  np.zeros(((H+1), 4))))


            for i in range(H+1):
                if i < H:
                    trajectory[i, 3:5] = trajectory[i+1, :2] - trajectory[i, :2]
                else:
                    trajectory[i, 3:5] = trajectory[i-1, 3:5]

                r, theta = polar(trajectory[i, 3:5])
                trajectory[i, 2] = (theta + 2*np.pi) % 2*np.pi

            # initialize optimization variables
            q_i = state
            q_d = trajectory
            #u = np.tile(self.u_max-self.u_min, H) * np.random.rand(H*3) + np.tile(self.u_min, H)

            # -----------------------------------------------
            
            point = transform_point(g, trajectory[-1, :2])
            d, phi = polar(point)

            steer = 0.25 * -phi
            accel = np.cos(phi)
            brake = 0.0

            u = np.array([steer, accel, brake])
            u = np.clip(u, self.u_min, self.u_max)
            u = np.tile(u, H)
            
            # -----------------------------------------------

            # bounds on control inputs
            bounds = (np.tile(self.u_min, H), np.tile(self.u_max, H))

            # nonlinear least-squares optimization
            solution = optimize.least_squares(J, u, args=(q_i, q_d, H), bounds=bounds, gtol=None, xtol=None)

            if not solution.success:
                print("[{}] optimization failed".format(FILE))

            # format and display solution
            self.u = solution.x.reshape((-1, 3))

        index = (t // T) % H
        return self.u[index]

    def model_predictive_path_integral_control(self, state, t):
        # TODO
        # x, y, theta, v_x, v_y, omega = state
        # alpha, beta, x, y = self.track[i]
        raise NotImplementedError

    def machine_learning_control(self, state, t):
        # TODO
        # x, y, theta, v_x, v_y, omega = state
        # alpha, beta, x, y = self.track[i]
        raise NotImplementedError

    def step(self, state, t):
        #self.action = self.random_control(state, t)
        #self.action = self.basic_control(state, t)
        self.action = self.model_predictive_control(state, t)
        #self.action = self.model_predictive_path_integral_control(state, t)
        #self.action = self.machine_learning_control(state, t)

        return self.action, self.done

class KeyboardController:
    def __init__(self, env):
        self.action = [0.0, 0.0, 0.0]
        self.done = False

        env.viewer.window.on_key_press = self.on_key_press
        env.viewer.window.on_key_release = self.on_key_release

    def on_key_press(self, k, mod):
        if k == key.LEFT:
            self.action[0] = -1.0

        if k == key.RIGHT:
            self.action[0] = +1.0

        if k == key.UP:
            self.action[1] = +1.0

        if k == key.DOWN:
            self.action[2] = +0.8

        if k == key.Q:
            self.done = True

    def on_key_release(self, k, mod):
        if k == key.LEFT  and self.action[0] == -1.0:
            self.action[0] = 0.0

        if k == key.RIGHT and self.action[0] == +1.0:
            self.action[0] = 0.0

        if k == key.UP:
            self.action[1] = 0.0

        if k == key.DOWN:
            self.action[2] = 0.0

    def step(self, state, t):
        return np.array(self.action), self.done

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

    def monitor_controller(self):
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
