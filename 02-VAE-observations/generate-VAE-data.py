import numpy as np
import random, tqdm
import multiprocessing as mp
import gym
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import CarRacing

_BATCH_SIZE = 128
_NUM_BATCHES = 20
_TIME_STEPS = 300
_RENDER = True

def generate_action(prev_action):
    if np.random.randint(3) % 3:
        return prev_action

    index = np.random.randn(3)
    # Favor acceleration over the others:
    index[1] = np.abs(index[1])
    index = np.argmax(index)
    mask = np.zeros(3)
    mask[index] = 1

    action = np.random.randn(3)
    action = np.tanh(action)
    action[1] = (action[1] + 1) / 2
    action[2] = (action[2] + 1) / 2

    return action*mask

def normalize_observation(obs):
    return obs.astype('float32') / 255.

def simulate_batch(batch_num):
    env = CarRacing()

    obs_data = []
    action_data = []
    action = env.action_space.sample()
    print(batch_num)
    for i_episode in range(_BATCH_SIZE):
        observation = env.reset()
        # Little hack to make the Car start at random positions in the race-track
        position = np.random.randint(len(env.track))
        env.car = Car(env.world, *env.track[position][1:4])
        observation = normalize_observation(observation)

        steps = 0
        obs_sequence = []
        action_sequence = []

        while steps < _TIME_STEPS:
            if _RENDER:
                env.render()

            steps += 1

            action = generate_action(action)

            obs_sequence.append(observation)
            action_sequence.append(action)

            observation, reward, done, info = env.step(action)
            observation = normalize_observation(observation)

        obs_data.append(obs_sequence)
        action_data.append(action_sequence)

        print("Batch {} Episode {} finished after {} timesteps".format(batch_num, i_episode, steps+1))
        print("Current dataset contains {} observations".format(sum(map(len, obs_data))))

    print("Saving dataset for batch {}".format(batch_num))
    np.save('./data/obs_data_VAE_{}'.format(batch_num), obs_data)
    np.save('./data/action_data_VAE_{}'.format(batch_num), action_data)

def main():
    print("Generating data for env CarRacing-v0")

    with mp.Pool(mp.cpu_count()) as p:
        tqdm.tqdm(p.map(simulate_batch, range(_NUM_BATCHES)), total=_NUM_BATCHES)

    env.close()

if __name__ == "__main__":
    main()
