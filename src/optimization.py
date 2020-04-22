import os

import numpy as np
from casadi import *

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

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

def J(z, u, z_ref, Q, R, H, model, dt):
    cost = 0

    for k in range(H):
        cost += (z[:, k] - z_ref[:, k]).T @ Q @ (z[:, k] - z_ref[:, k]) + u[:, k].T @ R @ u[:, k]

        z[:3, k+1] = z[:3, k] + z[3:, k]*dt
        z[2, k+1] = fmod(z[2, k+1], 2*np.pi)
        z[3:, k+1] = z[3:, k] + f(z[:, k], u[:, k], model)*dt

    cost += (z[:, H] - z_ref[:, H]).T @ Q @ (z[:, H] - z_ref[:, H])

    return cost

class NonlinearOptimizer():
    def __init__(self, model, dt):
        self.model = model
        self.dt = dt
        self.Q = MX(np.diag([1, 1, 1, 1, 1, 1]))
        self.R = MX(np.diag([1, 1, 1]))
        self.options = {'print_level': 0}

        self.options['mu_strategy'] = 'adaptive'
        self.options['max_iter'] = 1000
        self.options['tol'] = 1e-3
        #self.options['linear_solver'] = 'ma57'
        #self.options['ma57_automatic_scaling'] = 'yes'
        self.options['linear_scaling_on_demand'] = 'yes'
        self.options['hessian_approximation'] = 'limited-memory'
        self.options['limited_memory_update_type'] = 'bfgs'
        self.options['limited_memory_max_history'] = 10
        self.options['max_cpu_time'] = 1e8

    def run(self, z_init, u_init, z_ref, u_b, H):
        # CasADi Opti Stack
        opti = Opti()

        # optimization variable
        z = opti.variable(6, H+1)
        opti.set_initial(z, z_ref)

        # optimization variable
        u = opti.variable(3, H)
        opti.set_initial(u, u_init)

        # cost function
        opti.minimize(J(z, u, MX(z_ref), self.Q, self.R, H, self.model, self.dt))

        # constraints
        opti.subject_to(z[:, 0] == z_init)
        opti.subject_to(opti.bounded(u_b[0], vec(u), u_b[1]))

        # IPOPT
        opti.solver('ipopt', {'print_time': False}, self.options)

        try:
            sol = opti.solve()
            z_opt = sol.value(z)
            u_opt = sol.value(u)
        except RuntimeError:
            print('[{}] optimizer failed to find a solution'.format(FILE))
            z_opt = np.tile(z_init, (H+1, 1)).T
            u_opt = u_init

        return z_opt, u_opt
