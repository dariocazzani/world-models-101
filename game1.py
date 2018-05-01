"""
Solved Requirements
Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
"""

import numpy as np
import gym
import cma
import sys

def decide_action(observation, params):
    affine = np.dot(observation, params[:4]) + params[4]
    result = np.tanh(affine)
    if result > 0:
        return 1
    return 0

env = gym.make('CartPole-v0')

def play(params, render=False):
    observation = env.reset()
    score = []
    while True:
        if render:
            env.render()
        action = decide_action(observation, params)
        observation, reward, done, info = env.step(action)
        score.append(reward)
        if done:
            # print("Episode finished")
            break
    return -sum(np.array(score))

def train():
    es = cma.CMAEvolutionStrategy(5 * [0], 0.5)
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [play(x) for x in solutions])
    return es

if __name__ == '__main__':
    es = train()
    RENDER = False
    scores = []
    for _ in range(100):
        score = play(es.best.get()[0], render=RENDER)
        scores.append(-score)
        print(score)
    if np.mean(scores) >= 195:
        print("Solved")
    else:
        print("Not solved")
