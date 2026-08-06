from gym import Model  
import random as pyrandom
import pickle as pc
from numpy import *
import time

#kol = 100   
velicina_populacije = 10
elitizam = int(velicina_populacije * 0.1)
broj_generacija = 10
sigma = 0.5
populacija= []

maxi = -10000000

for i in range (velicina_populacije):
    x = Model(zeros(33600))
    populacija.append(x)

for _ in range(broj_generacija):
    print("generacija: ", _)
    t0 = time.time()
    evaluacija = []
    for model in populacija:
         score = model.eval_model()
         model.score = score
         evaluacija.append([score, model])
    t1 = time.time()
    print(f"eval phase: {t1-t0:.1f}s")

    #t2 = time.time()
    #print(f"full generation: {t2-t0:.1f}s", flush=True)

    evaluacija.sort(key=lambda item: item[0], reverse= True)
    if (evaluacija[0][0] > maxi):
        maxi = evaluacija[0][0]
        pc.dump(evaluacija[0][1], open('peakmodel.pkl','wb'))
        print("updated",'\n')
    print("score: ",evaluacija[0][0])
    print("br neurona: ", model.cnt)
    print("br veza: ", model.inv)
    nova_populacija = []

    for idx in range(elitizam):
        nova_populacija.append(evaluacija[idx][1])
    
    for i in range (velicina_populacije - elitizam):
        p1=pyrandom.choice(evaluacija[:int(velicina_populacije/3)])[1]
        p2=pyrandom.choice(evaluacija[:int(velicina_populacije/3)])[1]

        dijete = Model.__new__(Model)
        dijete.score = 0
        dijete.cross(p1,p2,sigma)
        nova_populacija.append(dijete)

    
    populacija = nova_populacija 
    sigma *= 0.995
    sigma = max(sigma, 0.02)    
    
    
