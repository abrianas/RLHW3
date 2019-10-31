from grid_world import *
import numpy as np
import scipy.signal
import gym
import pdb
import matplotlib.pyplot as plt
from mountain_car import *
N_POS = 15
N_VEL = 12

class DiscreteSoftmaxPolicy(object):
    def __init__(self, num_states, num_actions):
        self.num_states = num_states 
        self.num_actions = num_actions
        # here are the weights for the policy - you may change this initialization       
        self.weights = np.zeros((self.num_states, self.num_actions))


    # TODO: fill this function in    
    # it should take in an environment state
    def act(self, state):
        pass

    # TODO: fill this function in    
    # computes the gradient of the discounted return    
    # at a specific state and action    
    # use the computed advantage function appropriately.
    # return the gradient, a (self.num_states, self.num_actions) numpy array
    def compute_gradient(self, state, action, discounted_return,advantage):
        pass

    # TODO: fill this function in
    # takes a step of gradient ascent given a gradient (from compute_gradient())    
    # and a step size. adjust self.weights
    def gradient_step(self, grad, step_size):
        pass


class ValueEstimator(object):
    def __init__(self, num_states,num_actions):
        self.num_states = num_states
        self.num_actions = num_actions 
        
        #initial value estimates or weights of the value estimator are set to zero. 
        self.values = np.zeros((self.num_states))

    # TODO: fill this function in
    #takes in a state and predicts a value for the state
    def predict(self,state):
        pass

    # TODO: fill this function in
    # construct a suitable loss function and use it to update the 
    # values of the value estimator. choose suitable step size for updating the value estimator
    def update(self,state,value_estimate,target,value_step_size):
        pass


# TODO: fill this function in
# takes in a list of rewards from an episode
# and returns a list of discounted rewards
# Ex. get_discounted_returns([1, 1, 1], 0.5)
# should return [1.75, 1.5, 1]

def get_discounted_returns(rewards, gamma):
    pass

# TODO: fill this function in 
# this will take in an environment, GridWorld
# a policy (DiscreteSoftmaxPolicy)
# a value estimator,
# a discount rate, gamma
# and the number of episodes you want to run the algorithm for
# make sure to add in the baseline computation here. 
# Using the computed baseline, compute the advantage. 
# Use this advantage in the policy gradient calculation
def reinforce(env, policy,value_estimator):
        pass

    



if __name__ == "__main__":
    
    env = Continuous_MountainCarEnv()
    num_actions = 3
    num_states = N_POS * N_VEL #if you wish you can choose a different value.

    
    policy = DiscreteSoftmaxPolicy(num_states, num_actions)
    value_estimator = ValueEstimator(num_states, num_actions)
    reinforce(env, policy,value_estimator)

    #Test time
    state = env.reset()
    env.print()
    done = False
    while not done:
        input("press enter:")
        action = policy.act(state)
        state, reward, done = env.step(action)
        # env.print()


