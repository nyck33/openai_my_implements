#https://github.com/Hyeokreal/Actor-Critic-Continuous-Keras/blob/master/a2c_continuous.py
#Agent file for A2C continuous
import sys
import gym
import pylab
import numpy as np
from math import e #2.718281...
from scipy.stats import norm
from keras.layers import Dense, Input, Lambda
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K
import random
from collections import namedtuple, deque


from a2c_contin import Actor, Critic

ENV_ID = "Pendulum-v0"
GAMMA = 0.9
REWARD_STEPS = 2
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
ENTROPY_BETA = 1e-4

EPISODES = 10000

class A2CAgent:

    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        self.state_size = state_size
        self.action_size = action_size

        self.actor = Actor(self.state_size, self.action_size)

        self.critic = Critic(self.state_size, self.action_size)

        self.test = 1

        #self.actor_optimizer = actor_optimizer()
        #self.critic_optimizer = critic_optimizer()

        if self.load_model:
            self.actor.load_weights('pendulum_actor.h5')
            self.critic.load_weights('pendulum_critic.h5')

    #vs get_action for Pendulum using epsilon
    def act(self, state):
        state = np.reshape(state, [-1, self.state_size])
        mu, sigma_sq = self.actor.actor.predict(state)
        #sample action from normal distribution
        action = np.random.normal(mu, np.sqrt(sigma_sq))
        #just in case action is an extreme outlier beyond 68-95-99.7
        #clip it at the action range
        action = np.clip(action, -2, 2) #clip at [-2,2] for Pendulum
        return action

    def train_model(self, state, action, target, advantages):
        
        #TODO: must separate mu and sigma actors to train fit them????
        self.actor.optimizer([state, action, advantages])
        
        self.critic.optimizer([state, target])

if __name__ == "__main__":

    env = gym.make(ENV_ID)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    agent = A2CAgent(state_size, action_size)

    print(agent.test)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        mean = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            #reward /= 10

            next_state = np.reshape(next_state, [1, state_size])
            #TODO: why don't I need agent.critic.critic here?
            target, advantages = agent.critic.get_targets_adv(state, action, reward, next_state, done)

            agent.train_model(state, action, target, advantages)

            score += reward
            state = next_state

            if done:

                scores.append(score)
                episodes.append(e)
                mean = np.mean(scores[-min(10, len(scores)):])
                print("episode:", e, "  score:", score, "  mean:", mean)

                if np.mean(scores[-min(10, len(scores)):]) > -200:
                    sys.exit()
        if e % 50 == 0:
            agent.actor.actor.save_weights('pendulum_agent.h5')
            agent.critic.critic.save_weights('pendulum_critic.h5')