import sys
from numpy import *

rng = random.default_rng()

sys.path.pop(0)

import ale_py
import gymnasium as gym





gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5", obs_type="grayscale", render_mode="rgb_array")
env = gym.wrappers.FlattenObservation(env)


class model:
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
            biastemp = []
            for j in range (var):
                tempdublje = []
                for k in range (sloj[i]):
                    tempdublje.append(rng.random())
                temp.append(tempdublje)
                biastemp.append(rng.random())
            weights.append(temp)
            biases.append(biastemp)
        
        print(weights)
        #print(biases, '\n')
        #print(len(weights[0]))


        #return env.action_space.sample()

    

obs, info = env.reset()
while True:
    akcija = model([10,10,10,10,9], obs)
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs.shape, action, reward)
    print(obs)

    zb = matmul(l1,l2)
    print(zb)
    if terminated or truncated:
        break
env.close()








