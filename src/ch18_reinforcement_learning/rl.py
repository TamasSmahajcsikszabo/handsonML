'''
Reinforcement learning:
    - agent A takes actions in environment E after making observations
    - it receives rewards from the environment
    - agent learns to min-max pain-pleasure
    - the **algorithm** a software agent uses to determine its actions is
    called the policy: stochastic or deterministic policy
    - policy parameters: the number of parameters to finetune
    - policy search - finding the parameter values
    - policy space - the dimensions of policy parameters
    - one way is policy search
    - another way of search is by genetic algorithms
    - or by optimization (e.g. gradient ascent)
    - exploration / exploitation dilemma
    - discount factor
'''

# training an agent: Step 1. A working environment
# real or simulated
# for simulation: openAI gym
import gymnasium as gym
from gymnasium import Env
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


env = gym.make("CartPole-v1", render_mode="rgb_array")
env.reset(seed=42)
img = env.render()

env.action_space
# define next action
action = 1
# step executes a desired action
obs, reward, done, truncated, info = env.step(action)


def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1


totals = []
for episode in range(500):
    episode_reward = 0
    obs, info = env.reset(seed=episode)
    for step in range(200):
        action = basic_policy(obs)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        if done or truncated:
            break
    totals.append(episode_reward)

# use NN for training
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")  # 1 output with sigmoid
])

# credit assignment problem:
# when reward is received, which action should be credited for it or blamed for
# it


# more outputs would need softmax
# discount factor (gamma, DF) at each step:
# evaluate an action based on the sum of all the rewards that come after it
# DF: closer to 0 => future rewards count less
# typical DF is around 0.9 - 0.99
# return = sum of discounted rewards
# action advantage = estimation of how better an option is to other actions
# normalizing all the returns on each action


# policy gradiants (PG):
# PG alogrithms optimize the parameters of a policy by following the gradients
# toward a higher reward
# one popular class is the REINFORCE alogrithms

def play_one_step(env: Env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis])
        action = (tf.random.uniform([1, 1]) > left_proba)
        print(f"action: {action}")
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, truncated, info = env.step(int(action))
    return obs, reward, done, truncated, grads


def play_multiple_episodes(env: Env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs, info = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, truncated, grads = play_one_step(
                env, obs, model, loss_fn
            )
            current_rewards.append(reward)
            current_grads.append(grads)
            if done or truncated:
                break

        all_rewards.append(current_rewards)
        all_grads.append(current_grads)

    return all_rewards, all_grads


def discount_rewards(rewards, discount_factor):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_factor
    return discounted


def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [
        discount_rewards(
            rewards,
            discount_factor) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discount_rewards - reward_mean) / reward_std for discount_rewards in all_discounted_rewards]


# checking
discount_rewards([10, 0, -50], discount_factor=0.8)
discount_and_normalize_rewards([[10, 0, -50], [10, 20]], discount_factor=0.8)

# hyperparameters
n_iterations = 150
n_episodes_per_update = 10
n_max_steps = 200
discount_factor = 0.95
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01)
loss_fn = tf.keras.losses.binary_crossentropy

# training
for iteration in range(n_iterations):
    print(f"Iteration {iteration}")
    all_rewards, all_grads = play_multiple_episodes(
        env, n_episodes_per_update, n_max_steps, model, loss_fn
    )
    all_final_rewards = discount_and_normalize_rewards(
        all_rewards, discount_factor)

    all_mean_grads = []
    for var_index in range(len(model.trainable_variables)):
        mean_grads = tf.reduce_mean(
            [final_reward * all_grads[episode_index][step][var_index]
             for episode_index, final_rewards in enumerate(all_final_rewards)
             for step, final_reward in enumerate(final_rewards)],
            axis=0)
        all_mean_grads.append(mean_grads)

    optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))
