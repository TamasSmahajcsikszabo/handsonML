'''
Reinforcement learning:
    - agent A takes actions in environment E after making observations
    - it receives rewards from the environment
    - agent learns to min-max pain-pleasure
    - the algorithm a software agent uses to determine its actions is
    called the policy: stochastic or deterministic policy
    - policy parameters: the number of parameters to finetune
    - policy search - finding the parameter values
    - policy space - the dimensions of policy parameters
    - one way of search is by genetic algorithms
    - or by optimization
    - exploration / exploitation dilemma
'''

# training an agent: Step 1. A working environment
# real or simulated
# for simulation: openAI gym
import gym
import numpy as np
import tensorflow as tf

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
        print(episode_reward)
        if done or truncated:
            break
    totals.append(episode_reward)

# use NN for training
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid") #1 output with sigmoid
])

# more outputs would need softmax
# discout factor (gamma)
# action advantage = estimation of how better an option is to other actions
# policy gradiants (PG):
# PG alogirthms optimize the parameters of a policy by following the gradients
# toward a higher reward
# one popular class is the REINFORCE alogirthms
def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis])
        action = (tf.random.uniform([1,1]) > left_proba)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    gards = tape.gradient(loss, model.trainable_variables)
    obs, rewards, done, truncated, info = env.step(int(action))
    return obs, rewards, done, truncated, info



