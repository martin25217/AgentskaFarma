import sys
from numpy import *

rng = random.default_rng()

sys.path.pop(0)

import ale_py
import gymnasium as gym


gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5", obs_type="grayscale", render_mode = "human",repeat_action_probability=0.0)
env = gym.wrappers.FlattenObservation(env)


class Model:
    def cross(self, x1, x2, sigma):
        self.sigma = sigma

        weightx = []
        biasex = []

        for w1, w2 in zip(x1.weights, x2.weights):
            nw = (w1 + w2) / 2
            mask = rng.random(nw.shape) > 0.85
            nw = nw + mask * rng.normal(0, sigma, nw.shape)
            weightx.append(nw)

        for b1, b2 in zip(x1.biases, x2.biases):
            nb = (b1 + b2) / 2
            mask = rng.random(nb.shape) > 0.94
            nb += mask * rng.normal(0, sigma, nb.shape)
            biasex.append(nb)

        self.weights = weightx
        self.biases = biasex


    def feed_forward(self, ulaz):
        ulaz = ulaz / 127.5 - 1.0

        for weights, biases in zip(self.weights, self.biases):
            ulaz = tanh(matmul(ulaz, weights) + biases)

        return ulaz
    
    def __init__(self, sloj, ulaz):
        self.sloj = sloj
        self.ulaz = ulaz
        weights = []
        biases = []
        prev = ulaz
        for size in sloj:
            weights.append(rng.normal(0, 1/sqrt(prev), size=(prev, size)))
            biases.append(zeros(size))
            prev = size
        self.weights = weights
        self.biases = biases

 
    def eval_model(self):
        obs, info = env.reset()
        poc = info["lives"]
        result = 0
        prev_obs = obs.copy()
        while True:
            action = int(argmax(self.feed_forward(obs)))
            obs, reward, terminated, truncated, info = env.step(action)

            if reward == 0:
                reward-=1
            if action == 0:
                reward = -1000
            if action != 0 and array_equal(obs, prev_obs):
                reward = -1000
            if reward > 100:
               reward = 50

            prev_obs = obs.copy()
            result = result + reward
            if terminated or truncated or info["lives"] < poc:
                break
        return result
    
