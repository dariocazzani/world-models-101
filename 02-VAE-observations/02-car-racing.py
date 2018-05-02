"""
Solved Requirements
Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
"""

import numpy as np
import gym
import time
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import CarRacing

import cma
import multiprocessing as mp

from vae import load_vae

# Embdedded image vector: 32
# Num controllers: 3
# Bias: 1
_EMBEDDING_SIZE = 32
_NUM_ACTIONS = 3
_NUM_PARAMS = _NUM_ACTIONS * _EMBEDDING_SIZE + _NUM_ACTIONS

def normalize_observation(obs):
	return obs.astype('float32') / 255.

def get_weights_bias(params):
	weights = params[:_NUM_PARAMS - _NUM_ACTIONS]
	bias = params[-_NUM_ACTIONS:]
	weights = np.reshape(weights, [_EMBEDDING_SIZE, _NUM_ACTIONS])
	return weights, bias

def decide_action(sess, network, observation, params):
	observation = normalize_observation(observation)
	embedding = sess.run(network.z, feed_dict={network.image: observation[None, :,  :,  :]})
	weights, bias = get_weights_bias(params)

	action = np.matmul(np.squeeze(embedding), weights) + bias
	action = np.tanh(action)
	action[1] = (action[1] + 1) / 2
	action[2] = (action[2] + 1) / 2
	return action

env = CarRacing()

def play(params, render=True, verbose=True):
	sess, network = load_vae()
	observation = env.reset()
	# Little hack to make the Car start at random positions in the race-track
	np.random.seed(int(str(time.time()*1000000)[10:13]))
	position = np.random.randint(len(env.track))
	env.car = Car(env.world, *env.track[position][1:4])

	total_reward = 0.0
	steps = 0
	while True:
		if render:
			env.render()
		action = decide_action(sess, network, observation, params)
		observation, r, done, info = env.step(action)
		total_reward += r
		# NB: done is not True after 1000 steps when using the hack above for
		# 	  random init of position
		if verbose and (steps % 200 == 0 or steps == 999):
			print("\naction " + str(["{:+0.2f}".format(x) for x in action]))
			print("step {} total_reward {:+0.2f}".format(steps, total_reward))

		steps += 1
		if steps == 999:
			break

	return - total_reward

def train():
	es = cma.CMAEvolutionStrategy(_NUM_PARAMS * [0], 0.1, {'popsize': 56})
	try:
		while not es.stop():
			solutions = es.ask()
			with mp.Pool(mp.cpu_count()) as p:
				rewards = list(p.map(play, list(solutions)))
			print("rewards: {}".format(sorted(rewards)))
			es.tell(solutions, rewards)
	except (KeyboardInterrupt, SystemExit):
		print("Manual Interrupt")
	except Exception as e:
		print("Exception: {}".format(e))
	return es

if __name__ == '__main__':
	es = train()
	input("Press space to play... ")
	RENDER = True
	score = play(es.best.get()[0], render=RENDER, verbose=True)
	np.save('best_params', es.best.get()[0])
	print("Final Score: {}".format(-score))
