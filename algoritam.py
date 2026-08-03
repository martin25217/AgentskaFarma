from gym import Model  
import random 
import pickle as pc

kol = 30
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
    pc.dump(evaluacija[0][1], open('peakmodel.pkl','wb'))
    nova_populacija = []

    for idx in range(elitizam):
        nova_populacija.append(evaluacija[idx][1])
    
    for i in range (velicina_populacije - elitizam):
        p1=random.choice(populacija)
        p2=random.choice(populacija)

        dijete = Model([kol,kol,kol,kol,kol,9], 33600)
        dijete.cross(p1,p2,sigma)
        nova_populacija.append(dijete)

    print(populacija[0].eval_model())
    populacija = nova_populacija 
    
    
