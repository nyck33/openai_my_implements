import numpy as np
import gym
import pylab
import numpy as np
import tensorflow as tf

from keras import layers, models, optimizers
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU

import random
from collections import namedtuple, deque

class ReplayBuffer:
    """fixed-size buffer to store experience tuples"""

    def __init__(self, buffer_size, batch_size):
        """initialize a replaybuffer object.
        Params
        ======
            buffer_size:  max size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size) #100k
        self.batch_size = batch_size
        

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        "Randomly return a batch of experience from memory"
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)
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
        self.actor_target_lr= 0.0003
        self.build_model()

        
    def build_model(self):
        """Build actor(policy) network that maps states -> actions."""
        lrelu = LeakyReLU(alpha=0.1)
        #Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')


        #Add hidden layers, try different num layers, sizes, activation, batch_normalization
        #regularizers, etc
        net = layers.Dense(units=64, activation=lrelu)(states)
        net = layers.Dense(units=64, activation=lrelu)(net)
        #final layer with tanh which is a scaled sigmoid activation (between -1 and 1)
        #https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0
        raw_actions = layers.Dense(units=self.action_size, activation='tanh', name='raw_actions')(net)

        #Create keras model
        self.model = models.Model(input=states, outputs=raw_actions)
        #these lines are called from DDPG once the gradients of Q-value
        #obtained from critic
        #we define action_gradients which appears in K.function so it chains back from
        #K.function back up here I think
        #Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,)) #returns tensor

        loss = K.mean(-action_gradients * raw_actions)
        #####################################################
        #other losses here, ie. from regularizers
        #TODO: add entropy??? but that is OU noise

        #Define optimizer and training function for actor_local
        optimizer = optimizers.Adam(lr=self.actor_local_lr)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        #self.train_fn can be called in learn function
        self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()], \
        outputs=[], updates=updates_op)
        ##################################################

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
        self.critic_local_lr = 0.0001
        self.critic_target_lr = 0.0003

        self.build_model()

    #maps (state, action) pairs to Q values so map a tuple to a value?
    def build_model(self):
        lrelu = LeakyReLU(alpha=0.1)
        #Define input layers
        states = layers.Input(shape=(self.state_size,), name="states")
        actions = layers.Input(shape=(self.action_size,), name="actions")

        #Add hidden layers for state pathway
        net_states = layers.Dense(units=32, activation=lrelu)(states)
        net_states = layers.Dense(units=64, activation=lrelu)(net_states)

        #hidden layers for action
        net_actions = layers.Dense(units=32, activation=lrelu)(states)
        net_actions = layers.Dense(units=64, activation=lrelu)(net_actions)

        #try different layers sizes, activations, add batch normalization, regularizers, etc.

        #Combine state and action pathyways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation(lrelu)(net)

        #Add more layers ot the combined network if needed
            
        #Add final output layer to produce action values (Q-values)
        # array?
        Q_value = layers.Dense(units=1, name='q_values')(net)

        #Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_value)

        #Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.critic_local_lr)
        self.model.compile(optimizer=optimizer, loss='mse')

        #Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_value, actions)
        #self.get_action_gradients can be called
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()], outputs=action_gradients)

class OUNoise:
    """Ornstein-Uhlenbeck process"""

    def ___init__(self, size, mu, theta, sigma):
        """initialize parameters and noise process"""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """REset the internal state (=noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and returns as noise sample"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.signma * \
         np.random.randn(len(x))
        self.state = x + dx
        return self.state

