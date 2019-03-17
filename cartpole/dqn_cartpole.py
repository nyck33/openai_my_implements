#adjust leaky relu alpha, num hidden units 
#TODO: why epsilon starts at 0.999
import sys
import gym
import pylab
import random
import numpy as np
import datetime
from collections import deque
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 1000

class DQN_agent:
	def __init__(self, state_size, action_size):
		self.render= False
		self.load_model=False

		self.state_size = state_size
		self.action_size = action_size 
		#discount the TD target next q-value
		self.discount_factor =0.99
		self.learning_rate = 0.001

		self.epsilon=1.
		self.epsilon_decay = 0.999
		self.epsilon_min = 0.01
		#grab this many experiece tuples from memory
		self.batch_size = 64

		self.train_start = 1000

		self.memory=deque(maxlen=2000)

		self.model=self.build_model()
		self.target_model=self.build_model()

		self.update_target_model()

		if self.load_model:
			self.model.load_weights('./saved_models/cartpole_dqn.h5')

	def build_model(self):
		model = Sequential()
		#std distribution from [-limit, limit], limit=sqrt(6/num input units in weight tensor)
		model.add(Dense(8, input_dim=self.state_size, kernel_initializer='he_uniform'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(Dense(16, kernel_initializer='he_uniform'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
		model.summary()
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

		return model
	#update at set number of iterations of local model to avoid carrot-donkey, chasing moving target
	#look at formula where w is on both the left and right side of equation
	#updated after every episode
	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	def get_action(self, state, step):
		#e-greedy action
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		else: #exploit 
			q_value = self.model.predict(state)

			return np.argmax(q_value[0])#q-value is a list of lists

	def append_sample(self, state, action, reward, next_state, done, step):
		self.memory.append((state, action, reward, next_state, done))

		if(self.epsilon > self.epsilon_min) and (len(self.memory) > self.train_start):
			self.epsilon *= self.epsilon_decay

	def train_model(self):
		if len(self.memory) < self.train_start:
			return
		if (len(self.memory)==(self.train_start)):
			print("\n memory sufficient length, start training, epislon at {}".format(self.epsilon))
		#set batch size smaller of batch size or len memory
		batch_size = min(self.batch_size, len(self.memory))

		mini_batch = random.sample(self.memory, batch_size)

		input_states = np.zeros((batch_size, self.state_size))
		target_next_states = np.zeros((batch_size, self.state_size))
		action, reward, done = [], [], []

		for i in range(self.batch_size):#unpacking tuples into separate lists
			input_states[i] = mini_batch[i][0] #state
			action.append(mini_batch[i][1])
			reward.append(mini_batch[i][2])
			target_next_states[i] = mini_batch[i][3]
			done.append(mini_batch[i][4])

		#returns batch_size num q_values
		q_values = self.model.predict(input_states)

		#returns batch_size num target q_values from target network with stable weights
		target_qs = self.target_model.predict(target_next_states)

		for i in range(self.batch_size):
			if done[i]:
				q_values[i][action[i]] = reward[i] #q-value is reward if done
			else:#TD update target using greedy policy on target q values from target model, ie. stable weights
				q_values[i][action[i]] = reward[i] + self.discount_factor * (np.amax(target_qs[i]))

		#train local model on the updated TD target q_values for one epoch
		self.model.fit(input_states, q_values, batch_size=self.batch_size, epochs=1, verbose=0)

if __name__ == "__main__":

	env = gym.make('CartPole-v1')

	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n

	agent = DQN_agent(state_size, action_size)

	scores, episodes, means = [], [], []
	mean, std = 0, 0

	for episode in range(EPISODES):
		done = False
		score = 0
		state = env.reset()
		state = np.reshape(state, [1, state_size]) #flattened to 1 *4 array
		steps = 0

		while not done:
			if agent.render:
				env.render()
			steps+=1

			action=agent.get_action(state, steps)
			#either random e-greedy or greedy
			next_state, reward, done, info= env.step(action)
			next_state = np.reshape(next_state, [1, state_size])
			#-100 is arbitrary punishment for falling before terminal state of 500 steps
			#seems like it is overly focused on the final time step, ie. in vpg, I also
			#gave small punishments in rewards[-2:-5], ie. second to last to fifth to last
			reward = reward if not done or score == 499 else -100

			agent.append_sample(state, action, reward, next_state, done, steps)

			agent.train_model()

			score += reward
			state = next_state

			if done:
				agent.update_target_model()
				#add back 100 for book-keeping
				score = score if score == 500 else score +100
				scores.append(score)
				episodes.append(episode)
				mean = np.mean(scores[-10:])
				means.append(mean)
				std = np.std(scores[-10:])
				pylab.plot(episodes, scores, 'b', label='scores')#blue
				pylab.plot(episodes, means, 'g', label='mean')#green
				pylab.savefig("./save_graph/dqn_cartpole.png")
				if(episode % 1 == 0):
					print('episode: ', episode, ' score: ', score, " memory length: ", len(agent.memory), \
						" epislon: ", round(agent.epsilon, 4), " mean: ", round(mean, 2), "std: ", round(std, 2))
				#exit if last 10 scores over 490
				if np.mean(scores[-min(10, len(scores)):]) >= 490:
					sys.exit()

		if episode % 50 == 0:
			agent.model.save_weights('./saved_models/dqn_cartpole.h5')

