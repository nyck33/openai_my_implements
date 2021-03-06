# uses state values and not Q values as output of critic
"""
regularization in RL: https://arxiv.org/pdf/1810.00123.pdf
maximum entropy RL: https://medium.com/@awjuliani/maximum-entropy-policies-in-reinforcement-learning-everyday-life-f5a1cc18d32d
add entropy bonus
https://arxiv.org/abs/1704.06440
https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/

"""

import sys
import gym
import pylab
import numpy as np
from keras.layers import Dense, LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from keras import regularizers

EPISODES = 1000

# A2C(Advantage Actor-Critic) agent for the Cartpole


class A2CAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size  # is 4
        self.action_size = action_size  # is 2
        self.value_size = 1  # output size of critic

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99  # gamma for next state
        self.actor_lr = 0.001
        self.critic_lr = 0.005
        #self.actor_target_lr = 
        #self.critic_target_lr = 

        # create models 
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        
        #self.update_target_models()

        if self.load_model:
            self.actor.load_weights("./saved_models/cartpole_actor.h5")
            self.critic.load_weights("./saved_models/cartpole_critic.h5")

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu',
                  kernel_initializer='he_uniform' ))
        #actor.add(LeakyReLU(alpha=0.1))
        actor.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform'))
        actor.summary()
        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu',
                   kernel_initializer='he_uniform'))
        #critic.add(LeakyReLU(alpha=0.1))
        critic.add(Dense(self.value_size, activation='linear',
                         kernel_initializer='he_uniform'))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    
    # flatten the np array of probs
    def actor_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        # of action 1 or 2, return size 1 based on policy
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # update policy network every episode
    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))  # this is shape (1,1)?
        advantages = np.zeros((1, self.action_size)) #(1,2)

        Svalue = self.critic.predict(state)[0] #returns an array or tensor so must pick index
        next_Svalue = self.critic.predict(next_state)[0]

        if done:
            # only the chosen action gets a non-zero value in the flattened (1, 2) array
            # advantage of action = reward + ()
            #td error delta in () is a good estimator of advantage
            advantages[0][action] = reward - Svalue #because discount * next_Svalue = 0
            target[0][0] = reward
            #subtract from last 3 moves leading to error
            #nested loop
        else:
            advantages[0][action] = reward + (self.discount_factor * next_Svalue - Svalue)
            target[0][0] = reward + self.discount_factor * next_Svalue

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)

if __name__ == "__main__":
    
    env = gym.make('CartPole-v1')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = A2CAgent(state_size, action_size)

    scores, episodes, means, stds = [], [], [], [] 

    for episode in range(EPISODES):
        done = False
        score = 0
        mean, std, large = 0, 0, 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            # get action via stochastic policy
            action = agent.actor_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            reward = reward if not done or score == 499 else -100

            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:

                #agent.update_target_models()
                score = score if score == 500.0 else score +100
                scores.append(score)
                mean = np.mean(scores[-min(10, len(scores)):])
                means.append(mean) 
                std = np.std(scores[-min(10, len(scores)):])
                stds.append(std)
                large = max(scores)
                episodes.append(episode)
                pylab.plot(episodes, scores, 'b') #blue
                pylab.plot(episodes, means, 'g')
                pylab.plot(episodes, stds, 'r')
                pylab.savefig("./save_graph/cartpole_a2c.png")
                print('episode:', episode, 'score:', score, 'mean:', round(mean, 2), 'std:', round(std, 2), 'max:', large)
                # if last 10 scores mean over 490
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

        #save model
        if episode % 50 == 0:
            agent.actor.save_weights("./saved_models/cartpole_actor.h5")
            agent.critic.save_weights("./saved_models/cartpole_critic.h5")





