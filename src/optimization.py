import os

import numpy as np
import matplotlib.pyplot as plt
from casadi import *

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

def plot_solution(u_opt, z_init, z_ref, H, model, ts, border=0.1):
    plt.figure(num='solution', clear=True)
    
    z_opt = np.zeros((H+1, 6))
    z_opt[0] = z_init

    for k in range(H):
        f = model.predict([z_opt[k].reshape(1, -1), u_opt[k].reshape(1, -1)])[0]
        z_opt[k+1, :3] = z_opt[k, :3] + z_opt[k, 3:]*ts
        z_opt[k+1, 2] = z_opt[k+1, 2] % (2*np.pi)
        z_opt[k+1, 3:] = z_opt[k, 3:] + f*ts

    # control
    plt.subplot(2, 1, 1)
    plt.title('Model Predictive Control (H = {})'.format(H))
    plt.plot(range(1, H+1), u_opt[:, 0], '-o', color='C4', markersize=4, label='steering')
    plt.plot(range(1, H+1), u_opt[:, 1], '-o', color='C2', markersize=4, label='throttle')
    plt.plot(range(1, H+1), u_opt[:, 2], '-o', color='C3', markersize=4, label='brake')
    plt.xlabel('step')
    plt.xlim(1 - border, H + border)
    plt.ylim(-1 - border, 1 + border)
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
    plt.xlim(1 - border, H + border)
    plt.ylim(0 - border, 2*np.pi + border)
    plt.legend(loc='upper right')
    plt.grid()

    plt.pause(1e-3)

def f(z, u, model):
    tensor = vertcat(z, u)
    layers = model.get_weights()
    L = len(layers)

    for i in range(0, L, 2):
        weight, bias = MX(layers[i].T), MX(layers[i+1])

        if i < L-2:
            tensor = tanh(weight @ tensor + bias)
        else:
            tensor = weight @ tensor + bias

    return tensor

def J(u, z_init, z_ref, P, Q, R, H, model, ts):
    z = MX(6, H+1)
    z[:, 0] = z_init
    cost = 0

    for k in range(H):
        cost += (z[:, k] - z_ref[:, k]).T @ Q @ (z[:, k] - z_ref[:, k]) + u[:, k].T @ R @ u[:, k]

        z[:3, k+1] = z[:3, k] + z[3:, k]*ts
        z[3:, k+1] = z[3:, k] + f(z[:, k], u[:, k], model)*ts

    cost += (z[:, H] - z_ref[:, H]).T @ P @ (z[:, H] - z_ref[:, H])

    return cost

class NonlinearOptimizer():
    def __init__(self, model, ts=0.02):
        self.options = {'print_level': 0}
        self.model = model
        self.ts = ts

    def run(self, z_init, z_ref, u_init, u_min, u_max, P, Q, R, H):
        # CasADi Opti Stack
        opti = Opti()

        # optimization variable
        u = opti.variable(3, H)

        # initial value
        opti.set_initial(u, u_init)

        # cost function
        opti.minimize(J(u, MX(z_init), MX(z_ref), MX(P), MX(Q), MX(R), H, self.model, self.ts))
        
        # constraints
        opti.subject_to(opti.bounded(np.tile(u_min, H), vec(u), np.tile(u_max, H)))

        # IPOPT
        opti.solver('ipopt', {'print_time': False}, self.options)

        try:
            sol = opti.solve()
            u_opt = sol.value(u)
        except RuntimeError:
            print('[{}] optimizer failed to find a solution'.format(FILE))
            u_opt = u_init

        #plot_solution(u_opt.T, z_init, z_ref.T, H, self.model, self.ts)

        return u_opt
