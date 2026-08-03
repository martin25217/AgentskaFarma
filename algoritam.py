from gym import Model  
import random 
import pickle as pc

kol = 100
velicina_populacije = 100
elitizam = int(velicina_populacije * 0.1)
broj_generacija = 100
sigma = 0.5
populacija= []

maxi = 0

for i in range (velicina_populacije):
    x = Model([kol,kol,kol,kol,kol,9], 33600)
    populacija.append(x)

for _ in range(broj_generacija):
    evaluacija = []

    for model in populacija:
        score = model.eval_model()
        evaluacija.append([score, model])

    evaluacija.sort(key=lambda item: item[0], reverse= True)
    if (evaluacija[0][0] > maxi):
        maxi = evaluacija[0][0]
        pc.dump(evaluacija[0][1], open('peakmodel.pkl','wb'))
        print("updated",'\n')
    print("score: ",evaluacija[0][0])
    nova_populacija = []

    for idx in range(elitizam):
        nova_populacija.append(evaluacija[idx][1])
    
    for i in range (velicina_populacije - elitizam):
        p1=random.choice(populacija[30:])
        p2=random.choice(populacija[30:])

        dijete = Model([kol,kol,kol,kol,kol,9], 33600)
        dijete.cross(p1,p2,sigma)
        nova_populacija.append(dijete)

    
    populacija = nova_populacija 
    sigma *= 0.995
    sigma = max(sigma, 0.02)    
    
    
