#A2C continuous action space 
#TODO: list all inputs and outputs of each function
#TODO:  list the caller of each function so I have a UML type diagram
import sys
import gym
import pylab
import numpy as np
from math import e #2.718281...
from scipy.stats import norm
import keras
from keras.layers import Dense, Input, Lambda
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K
import random
from collections import namedtuple, deque

GAMMA = 0.9
REWARD_STEPS = 2
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
ENTROPY_BETA = 1e-4

class Actor:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.actor_lr = 0.0001
        self.hidden_size = 32

        self.actor = self.build_actor()

        self.optimizer = self.actor_optimizer()
        
    #input: state
    #output mu and sigma_sq
    def build_actor(self):

        #base for two heads of mean and variance
        base = Input(batch_shape=(None, self.state_size), name='states')
        net = Dense(units=self.hidden_size, use_bias=False, activation='relu')(base)

        #mu head
        mu = Dense(units=1, activation='tanh')(net)
        #custom layer for Pendulum
        mu = Lambda(lambda x: x * 2)(mu)

        #sigma head
        sigma_sq = Dense(units=1, activation='softplus')(net)
        #custom layer to ensure non-zero variance
        sigma_sq = Lambda(lambda x:x +0.0001)(sigma_sq)

        actor = Model(inputs=base, outputs=(mu, sigma_sq))

        #prep the function
        actor._make_predict_function()

        actor.summary()
        
        return actor
    #params: action, advantage
    #advantage calculated in critic get_targets_adv
    #optimizer instantiated in Agent.train_model 
    #use PDF to calculate loss
    #textbook uses sum of three losses
    def actor_optimizer(self):
        action = K.placeholder(shape=(None, 1))
        advantages = K.placeholder(shape=(None, 1))
        #self.model.outputs
        mu, sigma_sq = self.actor.output
        #mu, sigma_sq = self.actor.predict(state)
        #entropy of Gaussian
        entropy_loss = ENTROPY_BETA * (-K.mean(0.5 * (K.log(2. * np.pi * sigma_sq) + 1.)))
        
        #Prob Density Fn (PDF)
        #if sigma_sq is not None:
        #problem with clip, don't use TF tensor as bool error
            #sigma_sq = np.clip(sigma_sq,1e-3, None)
        p1 = - ((action - mu) ** 2) / (2 * K.clip(sigma_sq, 1e-3, None)) #clip min only
        p2 = - K.log(K.sqrt(2 * np.pi * sigma_sq))
        #log prob(a|s) given theta
        log_prob = p1 + p2
        #log_prob * score fn = advantage
        log_prob_v = advantages * log_prob
        loss_policy_v = -K.mean(log_prob_v) 
        #sum losses
        loss_v = loss_policy_v + entropy_loss 
        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss_v )
        train = K.function([self.actor.input, action, advantages], [], updates=updates)

        return train

class Critic:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
        self.hidden_size = 32
        self.critic_lr = 0.001
        self.critic = self.build_critic()
        self.optimizer = self.critic_optimizer()

    def build_critic(self):
        states = Input(batch_shape=(None, self.state_size,))

        net = Dense(units=self.hidden_size, activation='relu')(states)

        value = Dense(units=self.value_size, activation='linear')(net)
        #models.Model?
        critic = Model(inputs=states, outputs=value)

        #optimizer = optimizer.Adam(lr=self.critic_lr)
        critic.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.critic_lr))
        critic.summary()

        return critic

    def get_targets_adv(self, state, action, reward, next_state, done):
        #value_size = 1, action_size = 1
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))
        #why element [0]
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0] = reward - value #how much better is reward than expected value?
            target[0][0] = reward
        else:
            advantages[0] = reward + GAMMA * (next_value) - value
            target[0][0] = reward + GAMMA * next_value

        return target, advantages

    #need a critic optimizer and custom train fn to use target
    #minus value output of critic as base of MSE loss
    #from get_targets_adv
    
    def critic_optimizer(self):
        #placeholder for target???
        disc_reward = K.placeholder(shape=(None,1))
        #output of critic
        value = self.critic.output
        #MSE error
        loss = K.mean(K.square(disc_reward - value))

        optimizer = Adam(lr=self.critic_lr)
        #what is the second [] parameter???
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        #[] is an empty list for outputs?  https://www.tensorflow.org/api_docs/python/tf/keras/backend/function
        train = K.function([self.critic.input, disc_reward], [], updates=updates)
        return train
    

