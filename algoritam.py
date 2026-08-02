import sys
from numpy import *
import random

rng = random.default_rng()

def sigmoid(x):
    return 1 / (1 + exp(-x))

sys.path.pop(0)

import ale_py
import gymnasium as gym


gym.register_envs(ale_py)

populacija=[]
for i in range(101):
    populacija.append(Model([10,10,10,10,10,9], 33600))
    class Model:
    def feed_forward(self, ulaz):
        ulaz = ulaz / 255.0

        for weights, biases in zip(self.weights, self.biases):
            ulaz = sigmoid(matmul(ulaz, weights) + biases)

        return ulaz
    
    def __init__(self, sloj, ulaz):
        self.sloj = sloj
        self.ulaz = ulaz
        weights=[]
        biases=[]

        for i in range(len(sloj)):
            var = ulaz
            if (i > 0):
                var = sloj[i-1]
            
            temp = []
            
            for j in range (var):
                tempdublje = []
                for k in range (sloj[i]):
                    tempdublje.append(rng.random())
                temp.append(tempdublje)
                
            weights.append(temp)

        for i in range(len(sloj)):
            temp = []
            for j in range (sloj[i]):
                temp.append(rng.random())
            biases.append(temp)
        self.weights=weights
        self.biases=biases

    

    def eval_model(self):
        env = gym.make("ALE/MsPacman-v5", obs_type="grayscale", render_mode="rgb_array")
        env = gym.wrappers.FlattenObservation(env)
        obs, info = env.reset()
        result = 0
        while True:

            action = int(argmax(self.feed_forward(obs)))
            obs, reward, terminated, truncated, info = env.step(action)
            result = result + reward
    
            if terminated or truncated:
                break
        env.close()
        print(result)
        return result

