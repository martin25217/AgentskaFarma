from gym import Model  
import random 
import pickle as pc

kol = 2

populacija= []
for i in range (3):
    x = Model([kol,kol,kol,kol,kol,9], 33600)
    r1 = x.eval_model()
    populacija.append([r1,x])

populacija.sort(key=lambda item: item[0], reverse=True)

b=slice(10)
populacija = (sort[b])
for i in range (91):
    p1=random.choice(Model)
    p2=random.choice(Model)
    dijete.cross(p1,p2)
    populacija=populacija + dijete 









