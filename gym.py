import sys
from numpy import *

rng = random.default_rng()

def sigmoid(x):
    return 1 / (1 + exp(-x))

sys.path.pop(0)

import ale_py
import gymnasium as gym


gym.register_envs(ale_py)



class Model:
    def cross(self,x1,x2):
        self.x1 = x1
        self.x2 = x2
        weightx = []
        biasex = []
        for i in range(len(x1.weights)):
            temp = []
            for j in range(len(x1.weights[i])):
                tempdublje = []
                for o in range(len(x1.weights[i][j])):
                    nw = (x1.weights[i][j][o]+x2.weights[i][j][o])/2
                    tempdublje.append(nw)
                temp.append(tempdublje)
            weightx.append(temp)

        for i in range(len(x1.biases)):
            temp = []
            for j in range(len(x1.biases[i])):
                nw = (x1.biases[i][j]+x2.biases[i][j])/2
                temp.append(nw)
            biasex.append(temp)

        x1.weights = weightx
        x1.biases = biasex
        print("done", '\n')
        print(x1.weights[0][0][0])


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
        #print(weights, '\n')
        self.weights=weights
        self.biases=biases

    

    def eval_model(self):
        env = gym.make("ALE/MsPacman-v5", obs_type="grayscale", render_mode="human")
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
        return result