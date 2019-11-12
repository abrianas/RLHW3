from grid_world import *
import numpy as np
import scipy.signal
import gym
import pdb
import sys
import matplotlib.pyplot as plt
from mountain_car import *
N_POS = 15
N_VEL = 12

class DiscreteSoftmaxPolicy(object):
    def __init__(self, num_states, action_grid):
        self.num_states = num_states 
        self.action_grid = action_grid
        self.num_actions = len(self.action_grid)

        # here are the weights for the policy - you may change this initialization       
        #self.weights = np.zeros((self.num_states, self.num_actions))
        self.weights = np.random.random((self.num_states, self.num_actions))

    def softmax(self):
        num=np.exp(self.weights)
        pi=np.atleast_2d(1/np.sum(num,axis=1)).T*num
        return pi
        
    # TODO: fill this function in    
    # it should take in an environment state
    def act(self, state):
        pi=self.softmax()
        action_index = np.random.choice(self.num_actions,1,p=pi[state,:])
        return self.action_grid[action_index][0], action_index[0]

    # TODO: fill this function in    
    # computes the gradient of the discounted return    
    # at a specific state and action    
    # use the computed advantage function appropriately.
    # return the gradient, a (self.num_states, self.num_actions) numpy array
    def compute_gradient(self, state, action, discounted_return, advantage):
        pi=self.softmax()
        
        grad_left=np.zeros(self.weights.shape)
        grad_left[state,action]=1
        
        grad_right=np.zeros(self.weights.shape)
        grad_right[state,:]=pi[state,:]
        
        #grad_mat=(discounted_return)*(grad_left-grad_right)
        grad_mat=(advantage)*(grad_left-grad_right)
        return grad_mat

    # TODO: fill this function in
    # takes a step of gradient ascent given a gradient (from compute_gradient())    
    # and a step size. adjust self.weights
    def gradient_step(self, grad, step_size):
        self.weights = self.weights + grad*step_size


class ValueEstimator(object):
    def __init__(self, num_states):
        self.num_states = num_states
        
        #initial value estimates or weights of the value estimator are set to zero. 
        self.values = np.zeros((self.num_states))

    # TODO: fill this function in
    #takes in a state and predicts a value for the state
    def predict(self,state):
        value_predicted = self.values[state]
        return value_predicted

    # TODO: fill this function in
    # construct a suitable loss function and use it to update the 
    # values of the value estimator. choose suitable step size for updating the value estimator
    def update(self, state, value_estimate, target, value_step_size):
        gradientV = np.zeros((self.num_states)) # will store the gradient of value_estimate
        gradientV[state] = 1 # take the gradient of the state-value estimate
        #delta = abs(target-value_estimate)
        delta = (target-value_estimate)**2
        self.values += value_step_size*delta*gradientV 



# TODO: fill this function in
# takes in a list of rewards from an episode
# and returns a list of discounted rewards
# Ex. get_discounted_returns([1, 1, 1], 0.5)
# should return [1.75, 1.5, 1]
def get_discounted_returns(rewards, gamma):
    discount_rewards = np.zeros(len(rewards))
    temp = 0
    for t in reversed(range(0,len(rewards))):
        temp = temp*gamma+rewards[t]
        discount_rewards[t] = temp
    return discount_rewards

# TODO: fill this function in 
# this will take in an environment, GridWorld
# a policy (DiscreteSoftmaxPolicy)
# a value estimator,
# a discount rate, gamma
# and the number of episodes you want to run the algorithm for
# make sure to add in the baseline computation here. 
# Using the computed baseline, compute the advantage. 
# Use this advantage in the policy gradient calculation
def reinforce(env, policy, value_estimator, gamma, num_episodes, learning_rate, state_grid, base_flag):
    episode_rewards = []
    value_step_size = 0.00000005
    e=0
    while e<=num_episodes:
        state = env.reset()
        state = discretize(state, state_grid)

        episode_log = []
        iter = 0
        done = False

        while done == False and iter < 1000:
            # get action
            action, action_index = policy.act(state)
            
            # get next step, reward and chek whether reached goal
            next_state, reward, done, blah = env.step(action)
            
            # store the state, action, reward and next state
            episode_log.append([state, action_index, reward, next_state])

            state = discretize(next_state, state_grid)
            iter += 1


        episode_log = np.asarray(episode_log)
        rewards = episode_log[:,2]
        
        episode_rewards.append(np.sum(rewards))
        e+=1
        if done:           
            discount_rewards = get_discounted_returns(rewards, gamma)
            target = np.asarray(discount_rewards)
            for t in range(0,len(episode_log)):
                if base_flag: # reinforce with baseline
                    value_estimate = value_estimator.predict(episode_log[t,0])
                else: # reinforce w/o baseline
                    value_estimate = 0
                advantage = target[t] - value_estimate
                grads = policy.compute_gradient(episode_log[t,0], episode_log[t,1], discount_rewards[t],
                advantage)
                
                # update weight parameters
                value_estimator.update(episode_log[t,0], value_estimate, target[t], value_step_size)
                policy.gradient_step(grads, learning_rate)

    return episode_rewards
    
    
def discretize(state, grid):
    s = np.zeros([len(state),1])
    nX=len(grid[0])+1
    nV=len(grid[1])+1
    for l in range(0,len(grid)):

        s[l] = np.digitize(state[l], grid[0])

    ind=np.reshape(np.arange(nV*nX),[nX,nV])
    state_index=ind[int(s[0]),int(s[1])]
    return state_index
    
    
def create_grid(low, high, bins):
    grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] for dim in range(len(bins))]
    return grid

    

if __name__ == "__main__":
    gamma = 0.9
    num_episodes = 20000
    learning_rate = 0.0001
    env = Continuous_MountainCarEnv()
    num_actions = 3
    #num_states = N_POS * N_VEL # if you wish you can choose a different value.
    num_states = 9
    
    state_high = env.high_state
    state_low = env.low_state

    action_high = [env.max_action]
    action_low = [env.min_action]

    state_grid = create_grid(state_low, state_high, bins=(3,3))
    action_grid = np.linspace(action_low, action_high,3)

    # start with a new policy before calling reinforce
    policy = DiscreteSoftmaxPolicy(num_states, action_grid)
    value_estimator = ValueEstimator(num_states)
    episode_rewards2 = reinforce(env, policy, value_estimator, gamma, num_episodes, learning_rate, 
    state_grid, False)
    
    # start with a new policy before calling reinforce
    policy = DiscreteSoftmaxPolicy(num_states, action_grid)
    episode_rewards = reinforce(env, policy, value_estimator, gamma, num_episodes, learning_rate, 
    state_grid, True)
    
    # plot reinforce with and without baseline for comparison
    plt.plot(episode_rewards, label='reinforce w/ baseline')
    plt.plot(episode_rewards2, label='reinfoce w/o baseline')
    plt.xlabel("Number of Episodes")
    plt.ylabel("Total Rewards")
    plt.legend()
    plt.show()
    
    # # Re-train at least 5 times to provide an upper and lower limit for the observed reward 
    # # at each training step index
    # R=[]
    # for t in range(5):
        # policy = DiscreteSoftmaxPolicy(num_states, action_grid)
        # value_estimator = ValueEstimator(num_states)
        # r=reinforce(env, policy, value_estimator, gamma, num_episodes, learning_rate, state_grid, True)
        # R.append(r)
    
    # r_min=np.min(np.array(R),axis=0)
    # r_max=np.max(np.array(R),axis=0)
    # plt.figure()
    # plt.plot(r_min)
    # plt.plot(r_max)
    
    # plt.figure()
    # for i in range(5):
        # plt.plot(R[i])
    # plt.show()

