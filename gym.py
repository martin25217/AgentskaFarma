import pickle
import sys
from pathlib import Path

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

STOPA_MUTACIJE = 0.1


def mutiraj(x, korak):
    maska = rng.random(x.shape) < STOPA_MUTACIJE
    return x + maska * rng.normal(0, korak, x.shape)


def krizaj(a, b):
    return where(rng.random(a.shape) < 0.5, a, b)


class Model:
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump((self.sloj, self.ulaz, self.weights, self.biases), file)

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as file:
            sloj, ulaz, weights, biases = pickle.load(file)
        model = cls(sloj, ulaz)
        model.weights, model.biases = weights, biases
        return model

    def cross(self,x1,x2, sigma):
        self.weights = []
        self.biases = []
        for i, fan_in in enumerate(self.fan_in()):
            korak = sigma / sqrt(fan_in)
            self.weights.append(mutiraj(krizaj(x1.weights[i], x2.weights[i]), korak))
            self.biases.append(mutiraj(krizaj(x1.biases[i], x2.biases[i]), korak))

    def fan_in(self):
        return [self.ulaz if i == 0 else self.sloj[i-1] for i in range(len(self.sloj))]


    def feed_forward(self, ulaz):
        ulaz = ulaz / 127.5 - 1.0

        for weights, biases in zip(self.weights, self.biases):
            ulaz = tanh(matmul(ulaz, weights) + biases)

        return ulaz
    
    def __init__(self, sloj, ulaz):
        self.sloj = sloj
        self.ulaz = ulaz
        self.weights = []
        self.biases = []

        for i, var in enumerate(self.fan_in()):
            self.weights.append(rng.normal(0, 1 / sqrt(var), (var, sloj[i])))
            self.biases.append(zeros(sloj[i]))

    

    def eval_model(self, render_mode=None, seed=None):
        run_env = env
        if render_mode is not None:
            run_env = gym.wrappers.FlattenObservation(
                gym.make("ALE/MsPacman-v5", obs_type="grayscale", render_mode=render_mode)
            )

        obs, info = run_env.reset(seed=seed)
        poc = info["lives"]
        result = 0
        while True:

            action = int(argmax(self.feed_forward(obs)))
            obs, reward, terminated, truncated, info = run_env.step(action)
            result = result + reward

            if terminated or truncated or info["lives"] < poc:
                break

        if run_env is not env:
            run_env.close()
        return result
