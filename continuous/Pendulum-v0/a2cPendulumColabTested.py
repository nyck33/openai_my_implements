import sys
import gym
import numpy as np
from scipy.stats import norm
from keras.layers import Dense, Input, Lambda
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K

EPISODES = 3000



# A2C(Advantage Actor-Critic) agent for the Cartpole
class A2CAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        self.state_size = state_size #3
        self.action_size = action_size #1
        self.value_size = 1

        # get gym environment name
        # these are hyper parameters for the A3C
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.discount_factor = .9
        self.hidden1, self.hidden2 = 24, 24

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()

        # method for training actor and critic network
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        if self.load_model:
            self.actor.load_weights("./saved_models/PendulumK.h5")
            self.critic.load_weights("./saved_models/PendulumK.h5")

    def build_model(self):
        state = Input(batch_shape=(None, self.state_size))
        actor_input = Dense(30, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform')(state)
        # actor_hidden = Dense(self.hidden2, activation='relu')(actor_input)
        #tanh output of [-1,1]
        mu_0 = Dense(self.action_size, activation='tanh', kernel_initializer='he_uniform')(actor_input)
        #softplus gives output [0, inf] and deriv is sigmoid
        sigma_0 = Dense(self.action_size, activation='softplus', kernel_initializer='he_uniform')(actor_input)
        #mu is doubled to fit the action space of [-2, 2]?
        mu = Lambda(lambda x: x * 2)(mu_0)
        #custom layer ensures that sigma is not 0
        sigma = Lambda(lambda x: x + 0.0001)(sigma_0)
        #critic also takes in state and outputs a value
        critic_input = Dense(30, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform')(state)
        # value_hidden = Dense(self.hidden2, activation='relu')(critic_input)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(critic_input)

        actor = Model(inputs=state, outputs=(mu, sigma))
        critic = Model(inputs=state, outputs=state_value)
        #must declare before use
        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    def actor_optimizer(self):
        #placeholders for actions and advantages parameters coming in
        action = K.placeholder(shape=(None, 1))
        advantages = K.placeholder(shape=(None, 1))

        # mu = K.placeholder(shape=(None, self.action_size))
        # sigma_sq = K.placeholder(shape=(None, self.action_size))

        mu, sigma_sq = self.actor.output

        #defined a custom loss using PDF formula, K.exp is element-wise exponential
        pdf = 1. / K.sqrt(2. * np.pi * sigma_sq) * K.exp(-K.square(action - mu) / (2. * sigma_sq))
        #log pdf why?
        log_pdf = K.log(pdf + K.epsilon())
        #entropy looks different from log(sqrt(2 * pi * e * sigma_sq))
        #Sum of the values in a tensor, alongside the specified axis.
        entropy = K.sum(0.5 * (K.log(2. * np.pi * sigma_sq) + 1.))

        exp_v = log_pdf * advantages
        #entropy is made small before added to exp_v
        exp_v = K.sum(exp_v + 0.01 * entropy)
        #loss is a negation
        actor_loss = -exp_v

        #use custom loss to perform updates with Adam, ie. get gradients
        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
        #adjust params with custom train function
        train = K.function([self.actor.input, action, advantages], [], updates=updates)
        #return custom train function
        return train

    # make loss function for Value approximation
    def critic_optimizer(self):
        #placeholder for parameter target that comes in
        discounted_reward = K.placeholder(shape=(None, 1))
        #get output
        value = self.critic.output
        #MSE loss
        loss = K.mean(K.square(discounted_reward - value))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [], updates=updates)
        return train

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        mu, sigma_sq = self.actor.predict(np.reshape(state, [1, self.state_size]))
        # sigma_sq = np.log(np.exp(sigma_sq + 1))
        #return sample from std normal distribution 
        #epsilon is the random factor with mu and sigma_sq
        epsilon = np.random.randn(self.action_size)
        # action = norm.rvs(loc=mu, scale=sigma_sq,size=1)
        #mean + (std * epsilon)
        action = mu + np.sqrt(sigma_sq) * epsilon
        action = np.clip(action, -2, 2)
        return action

    # update policy network every episode
    #inputs of S,A,R,S'
    #potential output of advantages and target
    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        if done:
            advantages[0] = reward - value
            target[0][0] = reward
        else:
            advantages[0] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        #self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]
        #so [0] is tot train actor 
        self.optimizer[0]([state, action, advantages])
        #[1] trains critic
        self.optimizer[1]([state, target])


if __name__ == "__main__":
    # In case of CartPole-v1, maximum length of episode is 500
    env = gym.make('Pendulum-v0')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0] #3
    action_size = env.action_space.shape[0] #1

    # make A2C agent
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            reward /= 10
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100
            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                # every episode, plot the play time
                scores.append(score)
                episodes.append(e)
                print("episode:", e, "  score:", score)

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > -20:
                    sys.exit()

        # save the model
        if e % 50 == 0:
            agent.actor.save_weights("./saved_models/PendulumK_actor.h5")
            agent.critic.save_weights("./saved_models/PendulumK_critic.h5")