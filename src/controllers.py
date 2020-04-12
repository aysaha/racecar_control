import os
import time
import multiprocessing

import numpy as np
import inputs
from pyglet.window import key

from utils import *

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

def get_controller(ctrl, env):
    print("[{}] creating {} controller".format(FILE, ctrl))

    if ctrl == 'robot':
        controller = RobotController(env.viewer.window)
    elif ctrl == 'keyboard':
        controller = KeyboardController(env.viewer.window)
    elif ctrl == 'xbox':
        controller = XboxController()
    else:
        controller = None

    return controller

class RobotController():
    def __init__(self, window):
        self.done = False
        window.on_key_press = self.on_key_press
        
    def on_key_press(self, k, mod):
        if k == key.Q:
            self.done = True

    def step(self, state):
        raise NotImplementedError

        x, y, theta, v_x, v_y, omega = state

        return np.array([0.0, 0.0, 0.0]), self.done

class KeyboardController():
    def __init__(self, window):
        self.action = np.array([0.0, 0.0, 0.0])
        self.done = False

        window.on_key_press = self.on_key_press
        window.on_key_release = self.on_key_release

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

    def step(self, state):
        return self.action, self.done

class XboxController():
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
        while self.B.value == 0:
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

    def step(self, state):
        steer = np.clip(np.power(self.LeftJoystickX.value / XboxController.MAX_JOY_VAL, 3), -1, 1)
        accel = np.clip(np.power(self.RightTrigger.value / XboxController.MAX_TRIG_VAL, 2), 0, 1)
        brake = np.clip(np.power(self.LeftTrigger.value / XboxController.MAX_TRIG_VAL, 2), 0, 1)
        done = True if self.B.value == 1 else False
        return np.array([steer, accel, brake]), done
