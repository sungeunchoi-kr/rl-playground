##
# Q-function example with FrozenLake
#
# Taken from `https://www.kaggle.com/sarjit07/reinforcement-learning-using-q-table-frozenlake`
##

import gym
import torch
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense

import time
import matplotlib.pyplot as plt

## create environment (FrozenLake with is_slippery=False.)
from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
)

env = gym.make('FrozenLakeNotSlippery-v0')
# env = gym.make('FrozenLake-v1')

#
# Environment Notes:
# You cannot fall off the edge.
# If you fall into a hole, the game ends.
#
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# To check all environments present in OpenAI
# print(envs.registry.all())

num_episodes = 1000
steps_total = []
rewards_total = []
egreedy_total = []

# PARAMS 

# Discount on reward
gamma = 0.95

# Factor to balance the ratio of action taken based on past experience to current situtation
learning_rate = 0.9

# exploit vs explore to find action
# Start with 70% random actions to explore the environment
# And with time, using decay to shift to more optimal actions learned from experience

egreedy = 0.7
egreedy_final = 0.1
egreedy_decay = 0.999

# NN Model
model = Sequential()
model.add(InputLayer(batch_input_shape=(1, env.observation_space.n)))
model.add(Dense(20, activation='relu'))
model.add(Dense(env.action_space.n, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

for i_episode in range(num_episodes):

    # resets the environment
    state = env.reset()
    step = 0

    while True:
        step += 1

        if np.random.random() < egreedy:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax( model.predict(np.identity(env.observation_space.n)[state:state + 1]) )

        if egreedy > egreedy_final:
            egreedy *= egreedy_decay

        new_state, reward, done, info = env.step(action)

        # Filling the Q Table
        if done and reward < 1:
            target = 0
        else:
            target = reward + gamma * np.max( model.predict(np.identity(env.observation_space.n)[new_state:new_state + 1]))

        target_vector = model.predict(np.identity(env.observation_space.n)[state:state + 1])[0]

        print(f'state={state} action={action} target={target}')
        target_vector[action] = target

        model.fit(
          np.identity(env.observation_space.n)[state:state + 1],
          target_vector.reshape(-1, env.action_space.n),
          epochs=1, verbose=0)

        # Setting new state for next action
        state = new_state

        if done:
            steps_total.append(step)
            rewards_total.append(reward)
            egreedy_total.append(egreedy)
            if reward < 1.0:
                print('dropped off')
            else:
                print('reached the goal')

            if i_episode % 10 == 0:
                print('Episode: {} Reward: {} Steps Taken: {}'.format(i_episode,reward, step))
            break

model.save('model')

