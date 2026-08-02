import sys
from numpy import *

rng = random.default_rng()

def sigmoid(x):
    return 1 / (1 + exp(-x))

sys.path.pop(0)

import ale_py
import gymnasium as gym


gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5", obs_type="grayscale", render_mode="rgb_array")
env = gym.wrappers.FlattenObservation(env)


class model:
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
            var = len(ulaz)
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
            for j in range (len(sloj[i])):
                temp.append(rng.random())
            biases.append(temp)

    

obs, info = env.reset()
mreza = model([10,10,10,10,9], obs)
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs.shape, action, reward)
    print(obs)

    # zb = matmul(l1,l2)
    # print(zb)
    if terminated or truncated:
        break
env.close()








