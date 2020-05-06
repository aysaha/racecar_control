from abc import ABC, abstractmethod

import os
import imageio
import gym
import numpy as np
try:
    import roboschool   # noqa: F401
except ModuleNotFoundError:
    pass

import torch

from commons.utils import NormalizedActions, ReplayMemory

import sys
sys.path.append("../src")
from utils import *
from planning import plan_trajectory

def get_current_waypoint(trajectory, x, y):
    # waypoint horizon
    N = trajectory.shape[0]
    H = N // 100

    # create waypoint
    start = np.linalg.norm(trajectory[:, :2] - [x, y], axis=1).argmin()
    end = (start + H) % N
    waypoint = trajectory[end, :2]

    wp = trajectory[(end + 1) % len(trajectory), :3]
    wm = trajectory[(end - 1) % len(trajectory), :3]
    curvature = wp[2] - wm[2]  

    return list(waypoint) + [curvature]

def waypoint_to_car_ref(waypoint, x, y, theta):
    g = twist_to_transform([x, y, theta])
    g_inv = inv(g)
    point = transform_point(g_inv, waypoint[:2])
    return point

def speed_to_car_ref(v_x, v_y, theta):
    v_t = np.cos(theta) * v_x + np.sin(theta) * v_y
    v_n = - np.sin(theta)*v_x + np.cos(theta) * v_y
    return (v_t, v_n)


class AbstractAgent(ABC):

    def __init__(self, device, folder, config):

        self.folder = folder
        self.config = config
        self.device = device
        self.memory = ReplayMemory(self.config['MEMORY_CAPACITY'])
        #self.eval_env = NormalizedActions(gym.make(**self.config['GAME']))
        self.eval_env = gym.make('CarRacing-v1').env
        self.continuous = bool(self.eval_env.action_space.shape)

        self.state_size = 11 #self.eval_env.observation_space.shape[0]
        if self.continuous:
            self.action_size = self.eval_env.action_space.shape[0]
        else:
            self.action_size = self.eval_env.action_space.n

        self.display_available = 'DISPLAY' in os.environ

    @abstractmethod
    def select_action(self, state, episode=None, evaluation=False):
        pass

    def get_batch(self):

        transitions = self.memory.sample(self.config['BATCH_SIZE'])
        batch = list(zip(*transitions))

        # Divide memory into different tensors
        states = torch.FloatTensor(batch[0]).to(self.device)
        if self.continuous:
            actions = torch.FloatTensor(batch[1]).to(self.device)
        else:
            actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch[3]).to(self.device)
        done = torch.FloatTensor(batch[4]).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, done

    @abstractmethod
    def optimize(self):
        pass

    def evaluate(self, n_ep=2, render=False, gif=False):
        rewards = []
        if gif:
            writer = imageio.get_writer(self.folder + '/results.gif', duration=0.005)
        render = render and self.display_available

        try:
            for i in range(n_ep):
                state = self.eval_env.reset()
                trajectory = plan_trajectory(self.eval_env)
                reward = 0
                done = False
                steps = 0
                while not done and steps < self.config['MAX_STEPS']:
                    

                    #Get waypoint in the referential of the car
                    x, y, theta, v_x, v_y, omega = state[:6]
                    vt,vn = speed_to_car_ref(v_x, v_y, theta)
                    old_waypoint = get_current_waypoint(trajectory, x, y)

                    waypoint = waypoint_to_car_ref(old_waypoint[:2], x, y, theta)
                    modelstate = list(waypoint) + [old_waypoint[2], vt, vn, omega] + list(state[6:])
                    old_distance = np.linalg.norm(waypoint)
                    
                    #Selection action with respect to thais
                    action = self.select_action(modelstate, evaluation = True)

                    #Step through environment
                    next_state = self.eval_env.step(action) - action[0]**2 + np.sqrt(np.abs(vt))/10

                    #Get the new state and the distance between the old waypoint and the new position of the car
                    x, y, theta, v_x, v_y, omega = next_state[:6]
                    new_waypoint = waypoint_to_car_ref(old_waypoint[:2], x, y, theta) 
                    vt,vn = speed_to_car_ref(v_x, v_y, theta)
                    new_curvature = get_current_waypoint(trajectory, x, y)[2]
                    new_modelstate = list(new_waypoint) + [new_curvature, vt,vn, omega] +list(next_state[6:])
                    new_distance = np.linalg.norm(new_waypoint)

                    #print(v_x, v_y, vt, next_state[6])

                    #Compute the reward as how close to the old waypoint we are
                    r = - (new_distance/10)**2 - action[0]**2 + np.sqrt(np.abs(vt))/7

                    if render:
                        self.eval_env.render()
                    if i == 0 and gif:
                        writer.append_data(self.eval_env.render(mode='rgb_array'))
                    reward += r
                    steps += 1
                rewards.append(reward)

        except KeyboardInterrupt:
            if not render:
                raise

        finally:
            self.eval_env.close()
            if gif:
                print(f"Saved gif in {self.folder+'/results.gif'}")
                writer.close()

        score = sum(rewards)/len(rewards) if rewards else 0
        return score

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self, folder=None):
        pass
