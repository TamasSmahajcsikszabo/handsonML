'''
    The agent has only partial knowledge about the MDP
    - exploration policy: exploring the MDP and updating the state value
    estimates based on the transitions and rewards

    - Q-learning
    - the transition probs and rewards are initially unknown
    - watches an agent play e.g. randomly
    - gradually improves the estimates of the Q-values
    - chooses greedily the highest Q-value estimates
    Q(s, a) <- r + gamma * max( Q(s', a') )
    - for each pair of (s, a) this algorithm keeps track of a running average
    of the rewards r the agent gets upon leaving the state s with action a,
    plus the sum of discounteds future rewards it expects to get
    - off-policy algorithm


'''
import numpy as np
from markov import Q_values, transition_probabilities, rewards, possible_actions
from typing import Optional

'''
step function to take action and get reward
'''
def step(state, action):
    probas = transition_probabilities[state][action]
    next_state = np.random.choice([0, 1, 2], p=probas)
    reward = rewards[state][action][next_state]
    return next_state, reward

'''
Random policy
'''
def exploration_policy(state):
    return np.random.choice(possible_actions[state])

# hyperparameters
alpha0 = 0.05 # initial learning rate
decay = 0.005 # learning rate decay
gamma = 0.99 # DF
state = 0 # initial state
n_iterations = 10_000

for iteration in range(n_iterations):
    action = exploration_policy(state)
    next_state, reward = step(state, action)
    next_value = Q_values[next_state].max() # greedy
    alpha = alpha0 / (1 + iteration * decay)
    Q_values[state, action] *= 1 - alpha
    Q_values[state, action] += alpha * (reward + gamma * next_value)
    state = next_state

Q_values

'''
Epsilon-greedy policy:
    - at each step, it acts randomly with p of epsilon or greedily with
    1-epsilon p
    - it concentrates on the interesting parts of the environment
    Q(s, a) <- r + gamma * max f(Q(s', a'), N(s', a'))
    N(s', a') is the count how many times a' was chosen in state s'
    f(Q,N) is an exploration function: e.g. f(Q,N) = Q + kappa/(1+N)
    kappa is a curiosity hyperparameter: measurtes how much the agent is
    attracted to the unknown

Scaling:
    - Q-leanring doesn't scale well to medium or large MDP
    - Approximate QL: approximates the Q-value of any pair (s, a)
    using a managable number of parameters (parameter vector theta)
    such as: Q_theta(s,a)
    - linear combinations of features extracted from the state can be used
    - or recently DNN (deep neural networks)
    - Deep Q-Network (DQN): a DNN estimating Q-values: Deep Q-Learning
'''
