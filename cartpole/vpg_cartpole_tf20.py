#VPG Reinforce implementation 
#play out a full episode, assign positive to all moves in winning episodes
#and negative reward to all moves in losing episodes then update at end of each episode
#ideas taken from Karpathy and Silver plus my other forked Reinforcement repo

#pip install -q tensorflow==2.0.0-alpha0
import sys
import gym
import pylab
import numpy as np
import tensorflow as tf
from keras.layers import Dense, LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

#optional, can use tf.keras.layers.Dense, etc
#import keras

EPISODES = 1000

class ReinforceAgent:
	def __init__(self, state_size, action_size):

		self.render = False
		self.load_model = False
		#state and action size
		self.state_size = state_size #`env.observation_space.shape[0]`
		self.action_size = action_size #`env.action_space.n`

		#PG hyperparameters
		#discount subsequent rewards
		self.discount_factor = 0.99
		self.lr = 0.001
		#Karpathy had 200 neurons in the hidden layer
		self.hidden1, self.hidden2 = 64, 64
		droput = 0.5

		#create model for policy network
		self.model = self.build_model()

		#lists for collecting states, actions and rewards in episode
		self.states, self.actions, self.rewards = [], [], []

		#empty list for logprobs of +1 right or +1 left
		#2d list of log probs for each time step in episode
		self.logprobs = []

		if self.load_model:
			self.model.load_model("./saved_models/vpg_cartpole_tf20.h5")

	#approximate policy function using nn
	#state vector is input and prob of each action is output (push left or right)
	
	def build_model(self):
		model = Sequential()
		model.add(Dense(self.hidden1, input_dim=self.state_size))
		model.add(LeakyReLU(alpha=0.1))
		model.add(Dense(self.hidden2))
		model.add(LeakyReLU(alpha=0.1))
		model.add(Dense(self.action_size, activation='softmax'))

		model.summary()

		model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr))

		return model
	'''
	#want log probs for policy gradient: 
	#stackoverflow.com/questions/48465737/how-to-convert-log-probability-into-simple-probability-between-0-and-1-values-us
	def logprob(probs):
		return np.log(probs)
	'''
	def get_action(self, state):
		
		policy = self.model.predict(state, batch_size=1).flatten()
		#append the log probs for each action 
		probs = np.log(policy)
		self.logprobs.append(probs)
		return np.random.choice(self.action_size, 1, p=policy)[0] #returns array of 1, index 0 is the choice

	# In Policy Gradient, Q function is not available.
	# Instead agent uses sample discounted returns for evaluating policy
	def discount_rewards(self, rewards):
		discounted_rewards = np.zeros_like(rewards) #return array of zeros same shape and type 
		running_add = 0
		#print("\nrunnin_add initialized {}\n".format(running_add))
		#reversed starts from end of range, iterates towards 0
		for t in reversed(range(0, len(rewards))):
			running_add = running_add * self.discount_factor + rewards[t]
		 #   print("\nrunning add\n {}\n".format(running_add))
			discounted_rewards[t] = running_add
		#print("\ndiscounted_rewards\n {}\n".format(discounted_rewards))
		return discounted_rewards

	# save <s, a ,r> of each step in their own lists
	def append_sample(self, state, action, reward):
		self.states.append(state)
		#print("\nap self.states\n {}\n".format(self.states))
		self.rewards.append(reward)
		#print("\nap self.rewards\n {}\n".format(self.rewards))
		self.actions.append(action)
		#print("\nap self.actions\n {}\n".format(self.actions))
	# update policy network every episode at the end
	def train_model(self):
		episode_length = len(self.states) #list 1D
########################################################################################
#want to punish the last few moves leading to fall, not the entire episode
#episode: 126  score: 500.0  best: 500.0  mean: 181.09448818897638  std 163.59501720608517		
		if len(self.rewards) < 500:
			for i in range(len(self.rewards[-2 : -4])):
				self.rewards[-i - 1] = -3.
#########################################################################################				
		discounted_rewards = self.discount_rewards(self.rewards)
		
		discounted_rewards -= np.mean(discounted_rewards) #minus the mean
		
		discounted_rewards /= np.std(discounted_rewards)

		update_inputs = np.zeros((episode_length, self.state_size)) #no dtype= so this is 2D
		advantages = np.zeros((episode_length, self.action_size)) #2D
		
		for i in range(episode_length):
			#copying states array to update_inputs
			update_inputs[i] = self.states[i] #start from state zero
			advantages[i][self.actions[i]] = discounted_rewards[i]  

		#convert probs to log probs and negate (all log probs became sign-inverse of probs)
		#print("advantages\n {}".format(advantages))
		logprobs = np.array(self.logprobs) 
		#print('logprobs\n {}'.format(logprobs))
		
		labels = logprobs * advantages
		labels = np.negative(labels)
		#print('labels\n {}'.format(labels))

		self.model.fit(x=update_inputs, y=labels, epochs=1, verbose=0)

		self.states, self.actions, self.rewards, self.logprobs, = [], [], [], []

if __name__ == "__main__":

	env = gym.make('CartPole-v1')

	state_size = env.observation_space.shape[0]

	action_size = env.action_space.n

	agent = ReinforceAgent(state_size, action_size)

	scores, episodes = [], []

	best = 0

	for episode in range(EPISODES):
		done = False
		score = 0
		state = env.reset()
		#1D array of 4 floats for pole angle, move, cart dir and velocity
		state = np.reshape(state, [1, state_size]) 
		
		while not done:
			if agent.render:
				env.render()

			action = agent.get_action(state)

			next_state, reward, done, info = env.step(action)

			next_state = np.reshape(next_state, [1, state_size])
			#-100 for last stepif finish before terminal state of 
			#500 steps for training purposes
			#this is how it signals a losing episode with -100
			#TODO:  try to make all rewards -1 for losing instead
			reward = reward if not done or score == 499 else -100 

			agent.append_sample(state, action, reward)

			score += reward

			state = next_state

			if done:

				agent.train_model()

				#100 gets added back to losing episodes for total reward calculation
				score = score if score == 500 else score +100

				scores.append(score)
				
				if score >= max(scores):
					best = score

				mean = np.mean(np.array(scores))
				std = np.std(np.array(scores))
				episodes.append(episode)
				pylab.plot(episodes, scores, 'b')
				pylab.savefig('./save_graph/vpg_cartpole_tf20.png')
				print('episode:', episode, ' score:', score, ' best:', best, \
					' mean:', mean, ' std', std)

				if np.mean(scores[-min(10, len(scores)):]) > 490:
					sys.exit()
		if episode % 50 == 0:
			agent.model.save('./saved_models/vpg_cartpole_tf20.h5')

			


'''
after episode ends, need to give labels to for success and failure:
define success, if num_steps of current episode/mean(num_steps of ten previous episodes) == success
n-step update policy gradient
run model.fit() to get output probabilities, apply log and multiply by advantages
'''