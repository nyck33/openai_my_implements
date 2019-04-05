#DDPG implementation from Udacity Quadcopter fitted to gym
import numpy as np
import gym
import pylab
import numpy as np
import tensorflow as tf
import sys

from keras import layers, models, optimizers
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU

import random
from collections import namedtuple, deque
#import actor critic, replay, OU noise
from my_ddpg_ac import Actor, Critic, OUNoise

EPISODES = 1000

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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        

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

class DDPG():
    """Reinforcement learning agent that learns using DDPG"""
    #change from action size to action range I think
    def __init__(self, state_size, action_size, action_max):
        self.state_size = state_size
        self.action_size = action_size #1
        self.action_range = action_max * 2
        self.action_low = action_max - self.action_range
        self.action_high = action_max
        print("self.action_low", self.action_low) #-1
        print("self.action_high", self.action_high) #1
        #Actor Policy Model 
        self.actor_local = Actor(self.state_size, self.action_size, \
            self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, \
            self.action_low, self.action_high)

        #Critic Value Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        #initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        #Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, \
            self.exploration_theta, self.exploration_sigma)

        #replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        #Algorithm parameters
        self.gamma = 0.999 #discount parameter for critic TD error
        self.tau = 0.01 #soft update parameter
        """
        self.tau_actor = 0.1
        self.tau_critic = 0.5 
        """
        self.render = True
    #resets noise and env and outputs the initial state 
    def reset_episode(self):
        self.noise.reset()
        state = env.reset()
        self.last_state = state 
        return state

    #adds the experience to memory and calls learn fxn if memory sufficient for batch
    #
    def step(self, action, reward, next_state, done):
        self.memory.add(self.last_state, action, reward, next_state, \
            done)
       
        #Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
        #Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """Returns actions for given state(s) as per current policy"""
        state = np.reshape(state, [-1, self.state_size])
        raw_action = self.actor_local.model.predict(state)[0]
        noise = self.noise.sample()
        action = np.clip(raw_action + noise, -1, 1) #add noise for exploration
        return action 

    def learn(self, experiences):
        """update policy and value parameters using given batch of 
        experience tuples"""
        #convert experience tuples to separate arrays for each element
        #(states, actions, rewards, etc)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]) \
            .astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]) \
            .astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]) \
            .astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])               

        #get predicted next_state actions and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        #use the actions from target actor to find target Qs of next actions
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        #compute Q targets for current states using Q targets next from target_model
        #and train local critic model
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        """called to get dQ(s,a)/da from critic
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()], outputs=action_gradients)
        """
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), \
            (-1, self.action_size))

        """use the dQ(s,a)/da from critic to train actor
        called:  
        self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()], \
            outputs=[], updates=updates_op) no outputs
        """
        self.actor_local.train_fn([states, action_gradients, 1])

        #soft update target models, could use different tau for critic and actor
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

        #soft update target models
    def soft_update(self, local_model, target_model):
        """soft update model parameters"""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights)
        #weights to train target get updated using polyak formula where tau = [0,1]
        #closer to 1 than 0 usually
        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

if __name__== "__main__":

    env=gym.make("MountainCarContinuous-v0")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_bound = env.action_space.high
    #ensure action bound is symmetric
    assert (env.action_space.high == -env.action_space.low)

    agent = DDPG(state_size, action_size, action_bound)

    scores, episodes, means, stds = [], [], [], []

    position = 0

    for episode in range(EPISODES):
        done = False
        score = 0
        mean, std, large = 0, 0, 0
        state = agent.reset_episode()

        while not done:
            if agent.render:
                env.render()

            #get action
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            #adjust reward for incremental improvements in position
            #state[0] is the cart position from -1.2 to 0.6 
            reward += state[0] + 0.5
            #save experience in memory
            agent.step(action=action, reward=reward, next_state=next_state, done=done)
            
            #bookkeeping
            score += reward
            state = next_state

            if done:
#retype from here
                scores.append(score)
                mean = np.mean(scores[-min(10, len(scores)):])
                means.append(mean)
                std = np.std(scores[-min(10, len(scores)):])
                stds.append(std)
                large = max(scores)
                episodes.append(episode)
                pylab.plot(episodes, scores, 'b')
                pylab.plot(episodes, means, 'g')
                pylab.plot(episodes, stds, 'r')
                pylab.savefig("./saved_graphs/DDPGMountainCarContinuous.png")
                print('episode:', episode, 'score:', score, 'mean:', round(mean,2), 'std:', round(std, 2))
                if np.mean(scores[-min(100, len(scores)):]) >= 90.0:
                    sys.exit()
'''                    
        if episode % 50 == 0:
            agent.actor_local.save_weights("./saved_models/DDPGmtncarcont_actor.h5")
            agent.critic_local.save_weights("./saved_models/DDPGmtncarcont_critic.h5")
'''



    