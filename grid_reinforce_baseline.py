from grid_world import *
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import math


class DiscreteSoftmaxPolicy(object):
    def __init__(self, num_states, num_actions, temperature):
        self.num_states = num_states 
        self.num_actions = num_actions
        self.temperature = temperature
        
        # here are the weights for the policy - you may change this initialization       
        self.weights = np.zeros((self.num_states, self.num_actions))


    # TODO: fill this function in    
    # it should take in an environment state
    def act(self, state):
        pdf = np.zeros(self.num_actions)
        
        for a in range(self.num_actions):
        # soft max policy parameterization 
            pdf[a] = np.exp((self.weights[state][a])/self.temperature)/np.sum(np.exp(self.weights[state][:]/self.temperature))
  
        action = np.random.choice(self.num_actions,1,p=pdf)
        return action

    # TODO: fill this function in    
    # computes the gradient of the discounted return    
    # at a specific state and action    
    # use the computed advantage function appropriately.
    # return the gradient, a (self.num_states, self.num_actions) numpy array
    def compute_gradient(self, state, action, discounted_return, advantage): 
        grad = np.zeros((self.num_states, self.num_actions))
        pdf = np.zeros((self.num_actions)) # probability of getting action at a given state
        gradW = np.zeros(np.shape(self.weights))
        expected_value = np.zeros(np.shape(self.weights))
        
        # policy parameterization
        for a in range(self.num_actions):
            pdf[a] = np.exp(self.weights[state][a]/self.temperature)/np.sum(np.exp(self.weights[state][:]/self.temperature))
        
        # for taking the gradient of the weights w.r.t to weight[state][action]
        gradW[state][action] = 1
        expected_value[state][:] = pdf[:]
        grad = (gradW - expected_value)/self.temperature
        
        # incorporate the computed advantage
        gradient = advantage*grad
        return gradient        

    # TODO: fill this function in
    # takes a step of gradient ascent given a gradient (from compute_gradient())    
    # and a step size. adjust self.weights
    def gradient_step(self, grad, step_size):
        self.weights += step_size*grad


class ValueEstimator(object): ###
    def __init__(self, num_states):
        self.num_states = num_states
        
        # initial value estimates or weights of the value estimator are set to zero. 
        self.values = np.zeros((self.num_states))

    # TODO: fill this function in
    # takes in a state and predicts a value for the state
    def predict(self, state): ###
        value_predicted = self.values[state]
        return value_predicted

    # TODO: fill this function in
    # construct a suitable loss function and use it to update the 
    # values of the value estimator. choose suitable step size for updating the value estimator
    def update(self, state, value_estimate, target, value_step_size): ###
        gradientV = np.zeros((self.num_states)) # will store the gradient of value_estimate
        gradientV[state] = 1 # take the gradient of the state-value estimate
        #delta = abs(target-value_estimate)
        delta = (target-value_estimate)**2
        #print(target)
        #print(value_estimate)
        self.values += value_step_size*delta*gradientV 
        return


# TODO: fill this function in
# takes in a list of rewards from an episode
# and returns a list of discounted rewards
# Ex. get_discounted_returns([1, 1, 1], 0.5)
# should return [1.75, 1.5, 1]
def get_discounted_returns(rewards, gamma):
    discount = np.zeros(len(rewards))
    temp = 0
    for t in reversed(range(0,len(rewards))):
        temp = temp*gamma+rewards[t]
        discount[t] = temp
    return discount.tolist()

# TODO: fill this function in 
# this will take in an environment, GridWorld
# a policy (DiscreteSoftmaxPolicy)
# a value estimator,
# a discount rate, gamma
# and the number of episodes you want to run the algorithm for
# 
# make sure to add in the baseline computation here. 
# Using the computed baseline, compute the advantage
# Use this advantage in the policy gradient calculation
def reinforce(env, policy, value_estimator, gamma, num_episodes, learning_rate, base_flag):
    episode_rewards = []
    value_step_size = 0.0000005
    
    # Remember to update the weights after each episode, not each step
    for e in range(num_episodes):
        state = env.reset()
        episode_log = []
        rewards = []
        target = []
        score = 0
        
        done = False
        while True:
            # Sample from policy and take action in environment
            action = policy.act(state)
            next_state, reward, done = env.step(action)
            
            # Append results to the episode log
            episode_log.append([state, action, reward])
            
            
            # Save reward in memory for self.weights updates
            score += reward
            
            # If done, an episode has been complete, store the results for later
            if done:
                episode_log = np.array(episode_log)
                rewards = episode_log[:,2].tolist()
                discount = get_discounted_returns(rewards, gamma)
                target = np.asarray(discount) ###
                break
                
            state = next_state
            
        # Calculate the gradients and perform policy weights update
        for i in range(len(episode_log[:,0])):
            if base_flag: # reinforce with baseline
                value_estimate = value_estimator.predict(episode_log[i,0])
            else: # reinforce w/o baseline
                value_estimate = 0
            advantage = target[i] - value_estimate ###
            grads = policy.compute_gradient(episode_log[i,0], episode_log[i,1], discount[i],
                advantage)
            
            # update weight parameters
            value_estimator.update(episode_log[i,0], value_estimate, target[i], value_step_size) ###
            policy.gradient_step(grads, learning_rate)
            # the target is the estimate of the expectation of the Q-function
        
        # For logging the sum of the rewards for each episode
        episode_rewards.append(score)
        
    return episode_rewards    

    



if __name__ == "__main__":
    num_episodes = 20000
    learning_rate = 0.01
    gamma = 0.5
    temperature = 1 # temperature must be tuned.
    env = GridWorld(MAP2)
    env.print()
    
    # define the policy and the value estimator
    policy = DiscreteSoftmaxPolicy(env.get_num_states(), env.get_num_actions(), temperature)
    value_estimator = ValueEstimator(env.get_num_states())
    episode_rewards2 = reinforce(env, policy, value_estimator, gamma, num_episodes, learning_rate, False)
    value_estimator = ValueEstimator(env.get_num_states())
    policy = DiscreteSoftmaxPolicy(env.get_num_states(), env.get_num_actions(), temperature)
    episode_rewards = reinforce(env, policy, value_estimator, gamma, num_episodes, learning_rate, True)
    
    # Plot the training curves
    plt.plot(np.arange(num_episodes), episode_rewards, label='reinforce w/ baseline')
    plt.plot(np.arange(num_episodes), episode_rewards2, label='reinfoce w/o baseline')
    plt.xlabel("Number of Episodes")
    plt.ylabel("Total Rewards")
    plt.legend()
    plt.show()
    
    #Test time
    state = env.reset()
    env.print()
    done = False
    while not done:
        input("press enter:")
        action = policy.act(state)
        state, reward, done = env.step(action)
        env.print()
        
    # Runs reinforce algorithm 20 times training on 20,000 episodes each time 
    # and counts the number of times the goal is reached
    trials = 2
    for t in range(trials):
        print(t+1) # prints out the trial number
        num_goals = 0
        policy = DiscreteSoftmaxPolicy(env.get_num_states(), env.get_num_actions(), temperature)
        value_estimator = ValueEstimator(env.get_num_states())
        episode_rewards = reinforce(env, policy, value_estimator, gamma, num_episodes, learning_rate, True)
        for e in range(len(episode_rewards)):
            if episode_rewards[e] > 0:
                num_goals+=1
            else: # to look at policies that do not reach the goal
                print(episode_rewards[e])
                # state = env.reset()
                # env.print()
                # done = False
                # while not done:
                    # input("press enter:")
                    # action = policy.act(state)
                    # state, reward, done = env.step(action)
                    # env.print()
        print(num_goals)    


