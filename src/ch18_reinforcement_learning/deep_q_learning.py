'''
    - Q-leanring doesn't scale well to medium or large MDP
    - Approximate QL: approximates the Q-value of any pair (s, a)
    using a managable number of parameters (parameter vector theta)
    such as: Q_theta(s,a)
    - linear combinations of features extracted from the state can be used
    - or recently DNN (deep neural networks)
    - Deep Q-Network (DQN): a DNN estimating Q-values: Deep Q-Learning

    - computing:
        * given for pair (s, a) having computed the approximate Q-value by DQN
        * this estimated is aimed to be as close to r reward as possible
        after playing a action in state s, plus discounted future rewards
        * to compute this, we just estimate the sum of future discounted
        rewards for any s' and their any a' actions possible
        * we pick the highest estimates
        * by summin r reward and the future discounted value, we get a target
        Q-value estimate y(s, a)
        y(s, a) = r + gamma * max Q_theta(s', a')
        * with this target we run training with gradient descent
        * the goal is minimizing SE between estimated Q_theta(s, a) and y(s, a)
'''
from collections import deque
import numpy as np
import tensorflow as tf
import gymnasium as gym
from keras.api.losses import mean_squared_error

env = gym.make("CartPole-v1", render_mode="rgb_array")
env.reset(seed=42)


# step1:
# deep Q network:
# neural net, input: (s, a), output: approximate Q-value
# better in practive: input is any s, and output is Q-value approximations
# for all actions


input_shape = [4]
n_outputs = 2
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    tf.keras.layers.Dense(32, activation="elu"),
    tf.keras.layers.Dense(n_outputs)
])
# we will choose the action with the highest Q-value estimate
# we will use an epsilon greedy policy


def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        Q_values = model.predict(state[np.newaxis], verbose=0)[0]
        return Q_values.argmax()


# we will ise a replay buffer (replay memory)
replay_buffer = deque(maxlen=2000)


def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[index] for index in indices]
    return [np.array([experience[field_index] for experience in batch])
            for field_index in range(6)]


def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, truncated, info = env.step(action)
    replay_buffer.append((state, action, reward, next_state, done, truncated))
    return next_state, reward, done, truncated, info


batch_size = 32
discount_factor = 0.95
optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-2)
loss_fn = mean_squared_error


def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones, truncateds = experiences
    next_Q_values = model.predict(next_states, verbose=0)
    max_next_Q_values = next_Q_values.max(axis=1)
    runs = 1.0 - (dones | truncateds)
    target_Q_values = rewards + runs * discount_factor * max_next_Q_values
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


# training the model
for episode in range(600):
    obs, info = env.reset()
    for step in range(200):
        epsilon = max(1-episode/500, 0.001)
        obs, reward, done, truncated, info = play_one_step(env, obs, epsilon)
        if done or truncated:
            break

    if episode > 50:
        training_step(batch_size)

# catastrophic forgetting: as the agent explores the environment and learns
# updating its policy,
# it may learn new information in one part of the environment
#  that destroys what it has already learnt in other parts of the environment

'''
    Variants:
        1. fixed Q-valuie targets: to avoid loops, two DQNs are used
            - online model: learns and moves the agent
            - target model: sets the targets (this model is used to predict
            the next states' Q-values); it's updated less frequently

        2. Double DQN:
            - normally models, because they choose the max of Q-values, over
            estimate the Q- values
            - here the best action for the next state is selected by the online
            model
            - the target model only estimates the Q-values for the best action

        3. Importance Sampling (Prioritized Experience Replay):
            - instead of uniform sampling from the replay buffer
            - this focuses on important features
            - important experience = which leads to fast learning progress
            TD error: delta = r + gamma * V(s') - V(s)
            if large, it indicates a surprising transition
            this can be used to set priorities on the buffer
            - when an experience is sampled, it's p = | delta | is computed
            with a constant added
            - this p is proportional to p^zeta, where zeta is a greediness
            hyperparameter: 0 means uniform sampling, 1 full importance
            - downweighting is needed to avoid overfitting important
            experiences
            w = (n P)^-Beta
            n is the total number of experiences in the buffer
            Beta is the hyperparameter to compensate sampling frequency bias

        4. Dueling DQN:
            a Q-value of an (s, a) pair can be written as:
                Q(s,a) = V(s) + A(s, a)
                V(s) is the value of state s
                A(s, a) is the advantage of taking the action a in state s
                over to all other possible actions
                V(s) is equal to the Q-value of the best action a* for that state
                V(s) = Q(s, a*) in which case A(s, a*) = 0

'''
