"""
Solved Requirements
Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
"""

import numpy as np
import gym
import cma
import sys

def decide_action(params):
    params = np.tanh(params)
    params[1] = (params[1] + 1) / 2
    params[2] = (params[2] + 1) / 2
    return params

env = gym.make('CarRacing-v0')

def play(params, render=False):
    observation = env.reset()
    total_reward = 0.0
    steps = 0
    while True:
        if render:
            env.render()
        action = decide_action(params)
        s, r, done, info = env.step(action)
        total_reward += r
        if steps % 200 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in action]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))

        steps += 1
        if done:
            break
    return - total_reward

def train():
    es = cma.CMAEvolutionStrategy(3 * [0], 0.5)
    for _ in range(20):
    # while not es.stop():
        solutions = es.ask(16)
        es.tell(solutions, [play(x) for x in solutions])
    return es

if __name__ == '__main__':
    es = train()
    input("Press space to play... ")
    RENDER = True
    score = play(es.best.get()[0], render=RENDER)
    print("Final Score: {}".format(-score))
