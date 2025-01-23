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
