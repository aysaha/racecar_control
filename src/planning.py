import os

import numpy as np
import matlab

from learning import dynamics

FILE = os.path.basename(__file__)
DIRECTORY = os.path.dirname(__file__)

class OptimizationPlanner:
    def __init__(self, model, env):
        self.model = model
        self.track = np.array(env.track)
        self.q_min = np.array(env.observation_space.low)
        self.q_max = np.array(env.observation_space.high)
        self.u_min = np.array(env.action_space.low)
        self.u_max = np.array(env.action_space.high)
        self.engine = matlab.engine.start_matlab()

    def plot_plan(self):

        
        ax = plt.subplot(1, 1, 1)
        ax.set_aspect(1)
        ax.set_xlim(self.config_space.low_lims[0], self.config_space.high_lims[0])
        ax.set_ylim(self.config_space.low_lims[1], self.config_space.high_lims[1])

        for obs in self.config_space.obstacles:
            xc, yc, r = obs
            circle = plt.Circle((xc, yc), r, color='black')
            ax.add_artist(circle)

        if self.plan:
            plan_x = self.plan.positions[:, 0]
            plan_y = self.plan.positions[:, 1]
            ax.plot(plan_x, plan_y, color='green')

        plt.title('State Trajectory')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
        plt.show()
