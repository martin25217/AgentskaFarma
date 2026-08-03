from gym import Model  
import random 
#import pickle as pc
import numpy as np


def cross (parent1, parent2):
    for i in range (populacija-elitizam):
        dijete=Model([kol,kol,kol,kol,kol,9], 33600)
        return dijete

def pick_weighted(populacija, score, n=1):
    score = np.array(score, dtype=float)
    probabilities = score / score.sum()
    idx = np.random.choice(len(populacija), size=n, p=probabilities)
    return [populacija[i] for i in idx]


kol = 200

velicina_populacije = 30
elitizam = int(velicina_populacije * 0.1)
broj_generacija = 100
sigma = 1
populacija= []

for i in range (velicina_populacije):
    x = Model([kol,kol,kol,kol,kol,9], 33600)
    populacija.append(x)

for _ in range(broj_generacija):
    evaluacija = []

    for model in populacija:
        score = model.eval_model()
        evaluacija.append([score, model])

    evaluacija.sort(key=lambda item: item[0], reverse= True)
    
    nova_populacija = []

    for idx in range(elitizam):
        nova_populacija.append(evaluacija[idx][1])
    

    

    print(populacija[0].eval_model())
    populacija = nova_populacija 
    
    
