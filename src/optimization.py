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

def J(u, z_init, z_ref, P, Q, R, H, model, ts):
    z = z_init
    cost = 0

    for k in range(H):
        cost += (z - z_ref[:, k]).T @ Q @ (z - z_ref[:, k]) + u[:, k].T @ R @ u[:, k]

        z[:3] = z[:3] + z[3:]*ts
        z[3:] = z[3:] + f(z, u[:, k], model)*ts

    cost += (z - z_ref[:, H]).T @ P @ (z - z_ref[:, H])

    return cost

class NonlinearOptimizer():
    def __init__(self):
        self.u_opt = None
        self.options = {'print_level': 0, 'tol': 1e-3, 'max_iter': 1e3, 'max_cpu_time': 1e2}

        # default linear solver
        self.options['linear_solver'] = 'mumps'

        # faster linear solvers but requires external library (http://www.hsl.rl.ac.uk/ipopt/)
        #self.options['linear_solver'] = 'ma27' # small, serial
        #self.options['linear_solver'] = 'ma57' # small/medium, threaded BLAS
        #self.options['linear_solver'] = 'ma77' # huge, limited parallel
        #self.options['linear_solver'] = 'ma86' # large, highly parallelq
        #self.options['linear_solver'] = 'ma97' # small/medium/large, parallel

    def run(self, z_init, z_ref, u_min, u_max, H, model, ts):
        # gain matrices
        P = np.diag([1e3, 1e3, 1e3, 1e3, 1e3, 1e3])
        Q = np.diag([1e3, 1e3, 1e-9, 1e-9, 1e-9, 1e-9])
        R = np.diag([1e1, 1e1, 1e3])

        # initial guess
        if self.u_opt is not None:
            u_init = self.u_opt
        else:
            u_init = np.tile([[0.0], [0.5], [0.0]], (1, H))

        # CasADi Opti Stack
        opti = Opti()

        # optimization variable
        u = opti.variable(3, H)
        opti.set_initial(u, u_init)

        # cost function
        opti.minimize(J(u, MX(z_init), MX(z_ref), MX(P), MX(Q), MX(R), H, model, ts))

        # constraints
        opti.subject_to(opti.bounded(np.tile(u_min, H), vec(u), np.tile(u_max, H)))

        # IPOPT
        opti.solver('ipopt', {'print_time': False}, self.options)

        try:
            sol = opti.solve()
            self.u_opt = sol.value(u)
        except RuntimeError:
            print('[{}] optimizer failed to find a solution'.format(FILE))
            self.u_opt = np.tile([[0.0], [0.5], [0.0]], (1, H))

        return self.u_opt
