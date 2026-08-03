import sys
from numpy import *

rng = random.default_rng()

#def sigmoid(x):
   # return 1 / (1 + exp(-x))

sys.path.pop(0)

import ale_py
import gymnasium as gym


gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5", obs_type="grayscale", render_mode="rgb_array")
env = gym.wrappers.FlattenObservation(env)


class Model:
    def cross(self, x1, x2, sigma):
        self.x1 = x1
        self.x2 = x2
        self.sigma = sigma

        weightx = []
        biasex = []

        for w1, w2 in zip(x1.weights, x2.weights):
            nw = (w1 + w2) / 2
            mask = rng.random(nw.shape) > 0.9
            nw = nw + mask * rng.normal(0, sigma, nw.shape)
            weightx.append(nw)

        for b1, b2 in zip(x1.biases, x2.biases):
            biasex.append((b1 + b2) / 2)

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
        while True:

            action = int(argmax(self.feed_forward(obs)))
            obs, reward, terminated, truncated, info = env.step(action)
            result = result + reward
    
            if terminated or truncated or info["lives"] < poc:
                break
        env.close()
        return result
