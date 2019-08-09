import numpy as np
import gym
import copy
import pylab
import numpy as np
import tensorflow as tf

from keras import layers, models, optimizers
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Lambda
from keras.layers import GaussianNoise, Flatten, BatchNormalization

import random
from collections import namedtuple, deque


"""Params
    ======
        state_size(int): Dimension of each state
        action_size(float): continuous from -1 to 1
        Action range is +1 to -1
        ============
        action_low(float): Min value of each action dimension
        action_high(float): Max value of each action dim
    """

class Actor:
    """Policy model"""
    def __init__(self, state_size, action_size, action_low, action_high):
    
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.actor_local_lr = 0.0001
        self.actor_target_lr= 0.0003 #was 0.0003, tried 0.00015, next 0.00025
        self.build_model()

        
    def build_model(self):
        """Build actor(policy) network that maps states -> actions."""
        #Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')
        #Add hidden layers, try different num layers, sizes, activation, batch_normalization, regularizers, etc
        net = layers.Dense(units=128, )(states)
        #net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.GaussianNoise(1.0)(net)
        net = layers.Dense(units=256)(net)
        #net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.GaussianNoise(1.0)(net)
        net = layers.Dense(units=128)(net)
        #net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.GaussianNoise(1.0)(net)

        #final layer with tanh which is a scaled sigmoid activation (between -1 and 1)
        #https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0
        #raw_actions = layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)
        raw_actions = layers.Dense(units=self.action_size, activation='tanh', name='raw_actions')(net)
        
        #rescale to -2 and +2 with Lambda
        raw_actions = layers.Lambda(lambda x: x * 2.)(raw_actions)
        #Create keras model
        self.model = models.Model(input=states, outputs=(raw_actions))

        self.model.summary()
        """these lines are called from DDPG once the gradients of Q-value
        obtained from critic
        we define action_gradients which appears in K.function so it chains back from
        K.function back up here I think
        Define loss function using action value (Q value) gradients from critic
        placeholder below"""
        #scaled_actions = raw_actions * 2
        action_gradients = layers.Input(shape=(self.action_size,)) #returns tensor
        #rescale actions here to calculate loss (raw_actions *2)???????????????????????????
        loss = K.mean(-action_gradients * raw_actions)
        #####################################################
        #other losses here, ie. from regularizers
        #TODO: add entropy??? but that is OU noise

        #Define optimizer and training function for actor_local
        optimizer = optimizers.Adam(lr=self.actor_local_lr)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        """self.train_fn can be called in learn function
        call:
        self.actor_local.train_fn([states, action_gradients, 1])
        """
        self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()], \
            outputs=[], updates=updates_op)
        ##################################################
    #def save_weights():


"""Initialize parameters and build model.

    Params
    ======
        state_size(int): Dimension of each state
        action_size(int): Dimension of each action
        """
class Critic:

    def __init__(self, state_size, action_size):
    
        self.state_size = state_size
        self.action_size = action_size
        #initialize any other variable here
        self.critic_local_lr = 0.001 #taken from Emani DDPG
        self.critic_target_lr = 0.003 #was 0.003

        self.build_model()

    #maps (state, action) pairs to Q values so map a tuple to a value?
    def build_model(self):
        #lrelu = LeakyReLU(alpha=0.1)
        #Define input layers
        states = layers.Input(shape=(self.state_size,), name="states")
        actions = layers.Input(shape=(self.action_size,), name="actions")

        #Add hidden layers for state pathway
        net_states = layers.Dense(units=128)(states)
        #net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.GaussianNoise(1.0)(net_states)
        #net_states = layers.LeakyReLU(alpha=0.1)
        net_states = layers.Dense(units=128)(net_states)
        #net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.GaussianNoise(1.0)(net_states)

        #hidden layers for action
        net_actions = layers.Dense(units=128)(actions)
        #net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)
        net_actions = layers.GaussianNoise(1.0)(net_actions)
        
        net_actions = layers.Dense(units=128)(net_actions)
        #net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)
        net_actions = layers.GaussianNoise(1.0)(net_actions)

        #try different layers sizes, activations, add batch normalization, regularizers, etc.

        #Combine state and action pathyways
        net = layers.Add()([net_states, net_actions])
        #net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net) #was ReLU
        net = layers.GaussianNoise(0.5)(net)

        #Add more layers ot the combined network if needed
            
        #Add final output layer to produce action values (Q-values)
        # array?
        Q_values = layers.Dense(units=1, name='q_values')(net)

        #Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        #Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.critic_local_lr)
        self.model.compile(optimizer=optimizer, loss='mse')

        self.model.summary()

        #Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions) #dQ/dA
        """Q_values Tensor("q_values/BiasAdd:0", shape=(?, 1), dtype=float32) 
            actions Tensor("actions:0", shape=(?, 1), dtype=float32)"""
        print("Q_values", Q_values, "actions", actions) #
        print("action_gradients", action_gradients) #[None]
        print("action_gradients type", type(action_gradients)) #list
        #print("self.model.input", self.model.input) #states and actions
        
        """self.get_action_gradients can be called by agent's critic_local in def learn, action-grads used by action grads
        K.function runs the graph to get the Q-value necessary to calculate action_gradients which is the output
        call:
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), \
            (-1, self.action_size))
        """
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()], outputs=action_gradients)
        print("self.get_action_gradients", self.get_action_gradients)
"""
class OUNoise:
    #Ornstein-Uhlenbeck process

    def __init__(self, size, mu, theta, sigma):
        #initialize parameters and noise process
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        #REset the internal state (=noise) to mean (mu).
        self.state = copy.copy(self.mu)

    def sample(self):
        #Update internal state and returns as noise sample
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
         np.random.randn(len(x))
        self.state = x + dx
        return self.state
"""
