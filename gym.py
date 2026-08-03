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
    def cross(self,x1,x2, sigma):
        self.x1 = x1
        self.x2 = x2
        self.sigma = sigma
        weightx = []
        biasex = []
        for i in range(len(x1.weights):
            temp = []
            for j in range(len(x1.weights[i])):
                tempdublje = []
                for o in range(len(x1.weights[i][j])):
                    nw = (x1.weights[i][j][o]+x2.weights[i][j][o])/2
                    if(random.randint(0,100) > 90):
                        nw = nw + random.normal(0, sigma, 1)
                    print(flip)
                    tempdublje.append(nw)
                temp.append(tempdublje)
            weightx.append(temp)

        for i in range(len(x1.biases)):
            temp = []
            for j in range(len(x1.biases[i])):
                nw = (x1.biases[i][j]+x2.biases[i][j])/2
                temp.append(nw)
            biasex.append(temp)

        self.weights = [array(weight) for weight in weightx]
        self.biases = [array(bias) for bias in biasex]


    def feed_forward(self, ulaz):
        ulaz = ulaz / 127.5 - 1.0

        for weights, biases in zip(self.weights, self.biases):
            ulaz = tanh(matmul(ulaz, weights) + biases)

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
                    tempdublje.append(rng.normal(0, 1 / sqrt(var)))
                temp.append(tempdublje)
                
            weights.append(temp)

        for i in range(len(sloj)):
            temp = []
            for j in range (sloj[i]):
                temp.append(0.0)
            biases.append(temp)
        #print(weights, '\n')
        self.weights=[array(weight) for weight in weights]
        self.biases=[array(bias) for bias in biases]

    

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
