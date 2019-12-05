"""
Code taken from OpenAIGym
https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py

http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import numpy as np
from os import path

class Continuous_Pendulum:
    def __init__(self, g=10.0):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.g = g
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = np.array([self.max_torque, -self.max_torque])
        self.observation_space = np.array([-high, high])


        self.iter = 0
        self.maxiter = 200
        self.seed()

    def seed(self, seed=None):
        np.random.seed(seed)

    def step(self,u):
        th, thdot = self.state # th := theta

        g = self.g
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        self.iter += 1
        done = (self.iter == self.maxiter)
        return self._get_obs(), -costs, done, {}

    def reset(self):
        self.iter = 0
        high = np.array([np.pi, 1])
        self.state = np.random.uniform(low=-high, high=-high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


if __name__ == "__main__":
    print("hi")
    env = Continuous_Pendulum()
    env.reset()
    done = False
    action = np.array([0])
    rewards = []
    while not done:
        state, reward, done, _ = env.step(action)
        rewards.append(reward)

    print(len(rewards))
