"""
This implementation of PPO is based on a2c code with PPO subbed in for actor.

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
#from keras import regularizers
from keras import backend as K

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

        # old prob before update
        self.old_prob = 0.

        # epsilon value for clipping formula
        self.epsilon = 0.2

        # create models
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        # temp_actor
        self.temp_actor = self.build_actor()

        if self.load_model:
            self.actor.load_weights("./saved_models/cartpole_actor.h5")
            self.critic.load_weights("./saved_models/cartpole_critic.h5")

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model (policy)
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        # actor.add(LeakyReLU(alpha=0.1))
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
        # critic.add(LeakyReLU(alpha=0.1))
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

    # get targets and advantages
    def targets_and_adv(self, state, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))  # this is shape (1,1)?
        advantages = np.zeros((1, self.action_size))  # (1,2)

        Svalue = self.critic.predict(state)[0]  # returns an array or tensor so must pick index
        next_Svalue = self.critic.predict(next_state)[0]

        if done:
            # only the chosen action gets a non-zero value in the flattened (1, 2) array
            # advantage of action = reward + ()
            # td error delta in () is a good estimator of advantage
            advantages[0][action] = reward - Svalue  # because discount * next_Svalue = 0
            target[0][0] = reward
            # subtract from last 3 moves leading to error
            # nested loop
        else:
            advantages[0][action] = reward + (self.discount_factor * next_Svalue - Svalue)
            target[0][0] = reward + self.discount_factor * next_Svalue

        return target, advantages

    def train(self, state, action, reward, next_state, done, target, advantages):

        """Use predict method to get the current prob, use fit to train
        actor to get the new prob, then get new/old ratio and do the clip
        self.previous holds prob from before """

        """get the old aprob, action is already determined"""
        old_policy = self.actor.predict(state, batch_size=1).flatten()
        old_aprob = old_policy[action]
        """use temp_actor to get new prob so we don't update the actual actor until
        we do the clip op"""
        curr_weights = self.actor.get_weights()
        self.temp_actor.set_weights(curr_weights)
        self.temp_actor.fit(state, advantages, epochs=1, verbose=0)
        new_policy = self.temp_actor.predict(state, batch_size=1).flatten()
        new_aprob = new_policy[action]
        ########################stupid fucking Python
        ratio = new_aprob / old_aprob
        # scale = min(ratio * advantages, K.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages)
        no_clip = ratio * advantages
        clipped = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages

        self.actor.fit(state, np.minimum(no_clip, clipped), epochs=1, verbose=0)
        # clipping here, get weights and set on actor to update
        #clipped_advantage = clip(old_aprob, new_aprob, advantages)

        # we want to do below
        #self.actor.fit(state, clipped_advantage, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)
"""
    def clip(self, old_prob, new_prob, advs):
        ratio = new_prob / old_prob
        # scale = min(ratio * advantages, K.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages)
        no_clip = ratio * advs
        clipped = K.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advs
        if no_clip < clipped:
            return no_clip
        else:
            return clipped

        return None
"""
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

            target, advantages = agent.targets_and_adv(state, action, reward, next_state, done)

            agent.train(state, action, reward, next_state, done, target, advantages)

            score += reward
            state = next_state

            if done:

                # agent.update_target_models()
                score = score if score == 500.0 else score + 100
                scores.append(score)
                mean = np.mean(scores[-min(10, len(scores)):])
                means.append(mean)
                std = np.std(scores[-min(10, len(scores)):])
                stds.append(std)
                large = max(scores)
                episodes.append(episode)
                pylab.plot(episodes, scores, 'b')  # blue
                pylab.plot(episodes, means, 'g')
                pylab.plot(episodes, stds, 'r')
                pylab.savefig("./save_graph/cartpole_a2c.png")
                print('episode:', episode, 'score:', score, 'mean:', round(mean, 2), 'std:', round(std, 2), 'max:',
                      large)
                # if last 10 scores mean over 490
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

        # save model
        if episode % 50 == 0:
            agent.actor.save_weights("./saved_models/cartpole_actor.h5")
            agent.critic.save_weights("./saved_models/cartpole_critic.h5")
