"""
Solved Requirements
Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
"""

import numpy as np
import gym
import time, tqdm
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import CarRacing

import cma
import multiprocessing as mp

def decide_action(observation, params):
    params = np.tanh(params)
    params[1] = (params[1] + 1) / 2
    params[2] = (params[2] + 1) / 2
    return params

env = CarRacing()

def play(params, render=True, verbose=False):
    _NUM_TRIALS = 12
    agent_reward = 0
    for trial in range(_NUM_TRIALS):
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
            action = decide_action(observation, params)
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

        agent_reward += total_reward

    # If reward is out of scale, clip it
    agent_reward = np.maximum(-(100*_NUM_TRIALS), agent_reward)
    return - (agent_reward / _NUM_TRIALS)

def train():
    es = cma.CMAEvolutionStrategy(3 * [0], 0.1, {'popsize': 16})
    rewards_through_gens = []
    generation = 1
    try:
        while not es.stop():
            solutions = es.ask()
            with mp.Pool(mp.cpu_count()) as p:
                rewards = list(tqdm.tqdm(p.imap(play, list(solutions)), total=len(solutions)))

            es.tell(solutions, rewards)

            rewards = np.array(rewards) *(-1.)
            print("\n**************")
            print("Generation: {}".format(generation))
            print("Min reward: {:.3f}\nMax reward: {:.3f}".format(np.min(rewards), np.max(rewards)))
            print("Avg reward: {:.3f}".format(np.mean(rewards)))
            print("**************\n")

            generation+=1
            rewards_through_gens.append(rewards)
            np.save('rewards', rewards_through_gens)

    except (KeyboardInterrupt, SystemExit):
        print("Manual Interrupt")
    except Exception as e:
        print("Exception: {}".format(e))
    return es

if __name__ == '__main__':
    es = train()
    np.save('best_params', es.best.get()[0])
    input("Press space to play... ")
    RENDER = True
    score = play(es.best.get()[0], render=RENDER, verbose=True)
    print("Final Score: {}".format(-score))
