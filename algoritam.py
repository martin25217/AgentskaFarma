from gym import Model  
import random 
import pickle as pc

kol = 10
dijete = []
populacija= []
for i in range (101):
    populacija.append (Model([10,10,10,10,10,9], 33600))
sort=list(reversed(populacija))
b=slice(10)
populacija = (sort[b])
for i in range (91):
    p1=random.choice(Model)
    p2=random.choice(Model)
    dijete.cross(p1,p2)
    populacija=populacija + dijete 









