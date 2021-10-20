#!/usr/bin/env python3

import gym
import src.gym_gazebo.envs
import argparse
import time
import os
import pickle
import json
import random
import numpy as np
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout
from collections import deque

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, Permute, Input, MaxPooling2D, ELU, GRU, Reshape, ReLU
from tensorflow.keras.layers import TimeDistributed, BatchNormalization, Flatten, Add, Lambda, Concatenate, LayerNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.processors import MultiInputProcessor

from src.callbacks import *

import matplotlib.pyplot as plt
import sys
import signal

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*18)])
            tf.config.experimental.set_memory_growth(gpu, True)                     # Dynamic allocate GPU memory
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='GazeboEnv-v1')

args = parser.parse_args()

weight_path = "Model"
if not os.path.exists(weight_path):
    os.makedirs(weight_path)

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

WINDOW_LENGTH = 1

input_shape_img = (WINDOW_LENGTH,) + (48, 96)
input_shape_infos = (WINDOW_LENGTH,) + (364,)       # 360 + 4

processor = MultiInputProcessor(nb_inputs=2)

"""
Model
"""
img_input = Input(shape=input_shape_img, name='img_input')
x_img = Permute((2, 3, 1))(img_input)
x_img = Conv2D(filters=32, kernel_size=(4, 4), strides=(4, 4), activation='relu', input_shape=input_shape_img, data_format="channels_last")(x_img)
x_img = Conv2D(64, (3, 3), strides=(2, 2),  activation='relu', data_format="channels_last")(x_img)
x_img = Conv2D(64, (1, 1), strides=(1, 1),  activation='relu', data_format="channels_last")(x_img)
x_img = Flatten()(x_img)
x_img = Dense(128)(x_img)
x_img = Activation('relu')(x_img)

infos_input = Input(shape=input_shape_infos, name='infos_input')
# infos_input = Flatten()(infos_input)

lidar_infos, target_infos = tf.split(infos_input, num_or_size_splits=[360, 4], axis=2)
lidar_infos = Flatten()(lidar_infos)
target_infos = Flatten()(target_infos)

lidar_infos = Dense(180)(lidar_infos)
lidar_infos = Activation('relu')(lidar_infos)
lidar_infos = Dense(60)(lidar_infos)
lidar_infos = Activation('relu')(lidar_infos)

x_infos =  Concatenate()([lidar_infos, target_infos])   # 60 + 4

x_combine = Concatenate()([x_img, x_infos])     # 128 + 64
x_combine = Dense(128)(x_combine)
x_combine = Activation('relu')(x_combine)
x_combine = Dense(64)(x_combine)
x_combine = Activation('relu')(x_combine)
x_combine = Dense(nb_actions)(x_combine)
output_action = Activation('linear')(x_combine)

model = Model(inputs=[img_input, infos_input], outputs=output_action)
print(model.summary())

if args.mode == 'train':
    train = True
elif args.mode == 'test':
    train = False

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
try:
    memory = pickle.load(open(weight_path + "/memory.pkl", "rb"))
    print("Success to loaded memory!!!")
except (FileNotFoundError, EOFError):
    memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)                        #reduce memmory
    print("Fail to loaded memory!!!")

# Select a policy. We use eps-greedy action selection, which means that a random action is selected
# with probability eps. We anneal eps from 1.0 to 0.1 over the course of 1M steps. This is done so that
# the agent initially explores the environment (high eps) and then gradually sticks to what it knows
# (low eps). We also set a dedicated eps value that is used during testing. Note that we set it to 0.05c
# so that the agent still performs some random actions. This ensures that the agent cannot get stuck.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.0, value_min=.1, value_test=0.05,
                              nb_steps=100000)

"""
target_model_update >= 1 : Hard update every `target_model_update` steps.
0 <= target_model_update <= 1 : Soft update with `(1 - target_model_update) * old + target_model_update * new`.
"""
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, batch_size=32, processor=processor,
               nb_steps_warmup=5000, enable_double_dqn=True,
               enable_dueling_network=True, dueling_type='avg', train_interval=3,
               target_model_update=3000, policy=policy, gamma=.99)

dqn.compile(Adam(lr=0.001), metrics=['mae'])

if train:
    # Okay, now it time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.

    log_filename = weight_path + '/dqn_{}_log.json'.format(args.env_name)
    checkpoint_weights_filename = weight_path + '/dqn_' + args.env_name + '_weights_{step}.h5f'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=2000)]
    callbacks += [FileLogger(log_filename, interval=100)]

    if os.path.isfile(log_filename):
        model_index = 70000
        model_filename = weight_path + '/dqn_{}_weights_{}.h5f'.format(args.env_name, model_index)
        dqn.load_weights(model_filename)
        print("Weight loaded!!!")
    else:
        print("Weight not found!!!")

    # If `verbose` = 1, the number of steps('log_interval') that are considered to be an interval.
    train_history = dqn.fit(env, callbacks=callbacks, nb_steps=800100, visualize=False, verbose=2)
    print("Train Done!")

    # After training is done, we save the final weights.
    dqn.save_weights(weight_path + '/dqn_{}_weights.h5f'.format(args.env_name), overwrite=True)

    # Save memory
    pickle.dump(memory, open(weight_path + "/memory.pkl", "wb"))
    env.close()

else:

    test_model = [270000]
    for model_index in test_model:
        with open(weight_path + '/validation.txt', 'a') as fp:
            fp.write("Trained {} steps:\n".format(model_index))
        dqn.load_weights(weight_path + '/dqn_{}_weights_{}.h5f'.format(args.env_name, model_index))
        test_history = dqn.test(env, nb_episodes=250, visualize=True)
        print("Test Done!")
    env.close()
"""
==================================================================================================
"""
# LIVE_PLOT = False  # Rise a new window to plot process while training
# class Agent:
#     '''
#     Main class for agent
#     '''
#     def __init__(self, stateSize, actionSize):
#         self.isTrainActive = True  # Train model (Make it False for just testing)
#         self.loadModel = True  # Load model from file
#         self.loadEpisodeFrom = 8262  # Load Xth episode from file
#         self.episodeCount = 40000  # Total episodes
#         self.stateSize = stateSize  # Step size get from env, Observation Size
#         self.actionSize = actionSize  # Action size get from env
#         self.targetUpdateCount = 2000  # Update target model at every X step
#         self.saveModelAtEvery = 10  # Save model at every X episode
#         self.discountFactor = 0.99  # For qVal calculations
#         self.learningRate = 0.0003  # For neural net model
#         self.epsilon = 1.0  # Epsilon start value
#         self.epsilonDecay = 0.99  # Epsilon decay value
#         self.epsilonMin = 0.05  # Epsilon minimum value
#         self.batchSize = 64  # Size of a miniBatch
#         self.learnStart = 100000  # Start to train model from this step
#         self.memory = deque(maxlen=200000)  # Main memory to keep batches
#         self.timeOutLim = 1400  # Maximum step size for each episode
#         self.savePath = '/tmp/turtlebot3Model/'  # Model save path
#
#         self.onlineModel = self.initNetwork()
#         self.targetModel = self.initNetwork()
#
#         self.updateTargetModel()
#
#         # Create model file path
#         try:
#             os.mkdir(self.savePath)
#         except Exception:
#             pass
#
#     def initNetwork(self):
#         '''
#         Build DNN
#
#         return Keras DNN model
#         '''
#         model = Sequential()
#
#         model.add(Dense(64, input_shape=(self.stateSize,), activation="relu", kernel_initializer="lecun_uniform"))
#         model.add(Dense(64, activation="relu", kernel_initializer="lecun_uniform"))
#         model.add(Dropout(0.3))
#         model.add(Dense(self.actionSize, activation="linear", kernel_initializer="lecun_uniform"))
#         model.compile(loss="mse", optimizer=RMSprop(lr=self.learningRate, rho=0.9, epsilon=1e-06))
#         model.summary()
#
#         return model
#
#
#     def calcQ(self, reward, nextTarget, done):
#         """
#         Calculates q value
#         target = reward(s,a) + gamma * max(Q(s')
#
#         return q value in float
#         """
#         if done:
#             return reward
#         else:
#             return reward + self.discountFactor * np.amax(nextTarget)
#
#     def updateTargetModel(self):
#         '''
#         Update target model weights with online model weights
#         '''
#         self.targetModel.set_weights(self.onlineModel.get_weights())
#
#     def calcAction(self, state):
#         '''
#         Caculates an Action
#
#         returns action number in int
#         '''
#
#         if np.random.rand() <= self.epsilon:  # return random action
#             self.qValue = np.zeros(self.actionSize)
#             return random.randrange(self.actionSize)
#         else:  # Ask action to neural net
#             qValue = self.onlineModel.predict(state.reshape(1, self.stateSize))
#             self.qValue = qValue
#             return np.argmax(qValue[0])
#
#     def appendMemory(self, state, action, reward, nextState, done):
#         '''
#         Append state to replay mem
#         '''
#         self.memory.append((state, action, reward, nextState, done))
#
#     def trainModel(self, target=False):
#         '''
#         Train model with randomly choosen minibatches
#         Uses Double DQN
#         '''
#
#         # Get minibatches
#         miniBatch = random.sample(self.memory, self.batchSize)
#         xBatch = np.empty((0, self.stateSize), dtype=np.float64)
#         yBatch = np.empty((0, self.actionSize), dtype=np.float64)
#
#         for i in range(self.batchSize):
#             state = miniBatch[i][0]
#             action = miniBatch[i][1]
#             reward = miniBatch[i][2]
#             nextState = miniBatch[i][3]
#             done = miniBatch[i][4]
#
#             qValue = self.onlineModel.predict(state.reshape(1, len(state)))
#             self.qValue = qValue
#
#             if target:
#                 nextTarget = self.targetModel.predict(nextState.reshape(1, len(nextState)))
#             else:
#                 nextTarget = self.onlineModel.predict(nextState.reshape(1, len(nextState)))
#
#             nextQValue = self.calcQ(reward, nextTarget, done)
#
#             xBatch = np.append(xBatch, np.array([state.copy()]), axis=0)
#             ySample = qValue.copy()
#
#             ySample[0][action] = nextQValue
#             yBatch = np.append(yBatch, np.array([ySample[0]]), axis=0)
#
#             if done:
#                 xBatch = np.append(xBatch, np.array([nextState.copy()]), axis=0)
#                 yBatch = np.append(yBatch, np.array([[reward] * self.actionSize]), axis=0)
#
#         self.onlineModel.fit(xBatch, yBatch, batch_size=self.batchSize, epochs=1, verbose=0)
#
#
# class LivePlot():
#     '''
#     Class for live plot while training for episode and score
#     '''
#     def __init__(self):
#         self.x = [0]
#         self.y = [0]
#         self.fig = plt.figure(0)
#
#     def update(self, x, y, yTitle, text, updtScore=True):
#         if updtScore:
#             self.x.append(x)
#             self.y.append(y)
#
#         self.fig.canvas.set_window_title(text)
#         plt.xlabel('Epoch', fontsize=13)
#         plt.ylabel(yTitle, fontsize=13)
#         plt.style.use('Solarize_Light2')
#         plt.plot(self.x, self.y)
#         plt.draw()
#         plt.pause(0.5)
#         plt.clf()
#
#
# if __name__ == '__main__':
#     if LIVE_PLOT:
#         score_plot = LivePlot()
#
#
#     env = Turtlebot3GymEnv()  # Create environment
#
#     # get action and state sizes
#     stateSize = env.stateSize
#     actionSize = env.actionSize
#
#     # Create an agent
#     agent = Agent(stateSize, actionSize)
#
#     # Load model from file if needed
#     if agent.loadModel:
#         agent.onlineModel.set_weights(load_model(agent.savePath+str(agent.loadEpisodeFrom)+".h5").get_weights())
#
#         with open(agent.savePath+str(agent.loadEpisodeFrom)+'.json') as outfile:
#             param = json.load(outfile)
#             agent.epsilon = param.get('epsilon')
#
#
#     stepCounter = 0
#     startTime = time.time()
#     for episode in range(agent.loadEpisodeFrom + 1, agent.episodeCount):
#         done = False
#         state = env.reset()
#         score = 0
#         total_max_q = 0
#
#         for step in range(1,999999):
#             action = agent.calcAction(state)
#             nextState, reward, done = env.step(action)
#
#             if score+reward > 10000 or score+reward < -10000:
#                 print("Error Score is too high or too low! Resetting...")
#                 break
#
#             agent.appendMemory(state, action, reward, nextState, done)
#
#             if agent.isTrainActive and len(agent.memory) >= agent.learnStart:
#                 if stepCounter <= agent.targetUpdateCount:
#                     agent.trainModel(False)
#                 else:
#                     agent.trainModel(True)
#
#             score += reward
#             state = nextState
#
#             avg_max_q_val_text = "Avg Max Q Val:{:.2f}  | ".format(np.max(agent.qValue))
#             reward_text = "Reward:{:.2f}  | ".format(reward)
#             action_text = "Action:{:.2f}  | ".format(action)
#
#             inform_text = avg_max_q_val_text + reward_text + action_text
#
#             if LIVE_PLOT:
#                 score_plot.update(episode, score, "Score", inform_text, updtScore=False)
#
#             # Save model to file
#             if agent.isTrainActive and episode % agent.saveModelAtEvery == 0:
#                 weightsPath = agent.savePath + str(episode) + '.h5'
#                 paramPath = agent.savePath + str(episode) + '.json'
#                 agent.onlineModel.save(weightsPath)
#                 with open(paramPath, 'w') as outfile:
#                     json.dump(paramDictionary, outfile)
#
#             total_max_q += np.max(agent.qValue)
#
#             if (step >= agent.timeOutLim):
#                 print("Time out")
#                 done = True
#
#             if done:
#                 agent.updateTargetModel()
#
#                 avg_max_q = total_max_q / step
#
#                 # Infor user
#                 m, s = divmod(int(time.time() - startTime), 60)
#                 h, m = divmod(m, 60)
#
#                 print('Ep: {} | AvgMaxQVal: {:.2f} | CScore: {:.2f} | Mem: {} | Epsilon: {:.2f} | Time: {}:{}:{}'.format(episode, avg_max_q, score, len(agent.memory), agent.epsilon, h, m, s))
#
#                 if LIVE_PLOT:
#                     score_plot.update(episode, score, "Score", inform_text, updtScore=True)
#
#                 paramKeys = ['epsilon']
#                 paramValues = [agent.epsilon]
#                 paramDictionary = dict(zip(paramKeys, paramValues))
#                 break
#
#             stepCounter += 1
#             if stepCounter % agent.targetUpdateCount == 0:
#                 agent.updateTargetModel()
#
#         # Epsilon decay
#         if agent.epsilon > agent.epsilonMin:
#             agent.epsilon *= agent.epsilonDecay
#
