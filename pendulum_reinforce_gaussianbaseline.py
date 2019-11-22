import numpy as np
import scipy.signal

import pdb
import matplotlib.pyplot as plt
from pendulum import *
import pdb

class ContinuousPolicy(object):
    def __init__(self, num_states, num_actions, no_rbf):
        self.num_states = num_states
        self.num_actions = num_actions
        # here are the weights for the policy - you may change this initialization
        self.weights = np.zeros((no_rbf, self.num_actions))



    # TODO: fill this function in
    # it should take in an environment state
    def act(self, state):
        sigma = 0.9
        action = np.random.normal(np.dot(self.weights.T,state),sigma)

        return action

    # TODO: fill this function in
    # computes the gradient of the discounted return
    # at a specific state and action
    # use the computed advantage function appropriately.
    # return the gradient, a (self.num_states, self.num_actions) numpy array
    def compute_gradient(self, state, action,discount, advantage):

        sigma = 0.9

        grad = ((action - np.dot(self.weights.T,state))/sigma**2)*state
        pdb.set_trace()
        grad = advantage*grad

        return grad
    # TODO: fill this function in
    # takes a step of gradient ascent given a gradient (from compute_gradient())
    # and a step size. adjust self.weights
    def gradient_step(self, grad, step_size):
        #self.weights = self.weights + grad*step_size
        self.weights = self.weights + np.reshape(grad,[-1,1])*step_size


#design linear baseline
class LinearValueEstimator(object):
    def __init__(self, num_rbf):
        self.num_rbf = num_rbf
        self.weights_v = np.zeros(num_rbf)


    # TODO: fill this function in
    #takes in a state and predicts a value for the state
    def predict(self,state):
        ## here state is the rbf feacture vector

        value_predicted  = np.matmul(state,self.weights_v.T)
        return value_predicted

    # TODO: fill this function in
    # construct a suitable loss function and use it to update the
    # values of the value estimator. choose suitable step size for updating the value estimator
    def update(self, state, value_estimate, target, value_step_size):

        delta = target - value_estimate
        gradientV = state

        self.weights_v = self.weights_v + value_step_size*delta*gradientV


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
# a continuous policy
# a value estimator,
# a discount rate, gamma
# and the number of episodes you want to run the algorithm for
# make sure to add in the baseline computation here.
# Using the computed baseline, compute the advantage.
# Use this advantage in the policy gradient calculation
def reinforce(env, policy, value_estimator, gamma, num_episodes,value_step_size, learning_rate,base_flag):

    episode_rewards = []


    for e in range(num_episodes):
        print(e)
        state = env.reset()
        state = compute2dstate(state)

        feature_state = rbf(state, centers, rbf_sigma)

        episode_log = []
        iter = 0
        done = False

        while done == False and iter < 1000:
            # get action

            action = policy.act(feature_state)

            # get next step, reward and chek whether reached goal
            next_state, reward, done, blah= env.step(action)
            next_state = compute2dstate(next_state)

            # store the state, action, reward and next state
            episode_log.append([feature_state, action, reward, next_state])

            feature_state = rbf(next_state, centers, rbf_sigma)
            iter += 1


        episode_log = np.asarray(episode_log)
        rewards = episode_log[:,2]
        episode_rewards.append(np.sum(rewards))

        if done:


            discount_rewards = get_discounted_returns(rewards, gamma)
            target = np.asarray(discount_rewards)
            for t in range(0,len(episode_log)):
                if base_flag: # reinforce with baseline
                    value_estimate = value_estimator.predict(episode_log[t,0])
                else: # reinforce w/o baseline
                    value_estimate = 0
                advantage = target[t] - value_estimate
                grads = policy.compute_gradient(episode_log[t,0], episode_log[t,1], discount_rewards[t],advantage)

                # update weight parameters
                value_estimator.update(episode_log[t,0], value_estimate, target[t], value_step_size)
                policy.gradient_step(grads, learning_rate)

    return episode_rewards

def compute2dstate(state):
    theta = np.arctan(state[1]/state[0])
    state_2d = np.array([theta, state[2]])
    return state_2d


def rbf(state, centers, rbf_sigma):
    ## input state, return rbf features pi
    phi = []
    for c in range(0, len(centers)):
        rbf_eval =  np.exp(-np.linalg.norm(state - centers[c,:])**2/(2*(rbf_sigma**2)))
        phi.append(rbf_eval)
    return np.asarray(phi)

def computeRBFcenters(th_low, th_high, th_dot_low, th_dot_high, no_rbf):
    theta = np.linspace(th_low, th_high, no_rbf)
    thetadot = np.linspace(th_dot_low, th_dot_high, no_rbf)
    theta_c, thetadot_c = np.meshgrid(theta, thetadot)
    centers = []

    for i in range(0,no_rbf):
        for j in range(0,no_rbf):
            c = [theta_c[i,j], thetadot_c[i,j]]
            centers.append(c)

    centers = np.asarray(centers)
    return centers


if __name__ == "__main__":
    env = Continuous_Pendulum()
    num_episodes = 5000
    gamma = 0.9
    learning_rate = 0.01
    value_step_size = 0.005

    th_low = -np.pi
    th_high = np.pi
    th_dot_low = -8.0
    th_dot_high = 8.0
    no_rbf = 4
    rbf_sigma = 1.0/(no_rbf - 1)
    centers = computeRBFcenters(th_low, th_high, th_dot_low, th_dot_high, no_rbf)
    no_centers = len(centers)



    # TODO: define num_states and num_actions
    policy = ContinuousPolicy(2,1,no_centers)
    value_estimator = LinearValueEstimator(no_centers)
    reinforce(env, policy, value_estimator, gamma, num_episodes,value_step_size, learning_rate,True)

    # Test time
    state = env.reset()
    state = compute2dstate(state)
    feature_state = rbf(state, centers, rbf_sigma)
    # env.print()
    done = False
    state_hist = []
    while not done:
        action = policy.act(feature_state)

        state, reward, done, blah = env.step(action)
        state_hist.append(state)
        state = compute2dstate(state)
        feature_state = rbf(state, centers, rbf_sigma)

    # Plotting test time results
    state_hist = np.array(state_hist)
    plt.plot(state_hist[0, :])
    plt.xlabel("time (s)")
    plt.ylabel("angle (rad)")
    plt.show()
