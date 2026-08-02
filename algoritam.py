from gym import Model  
import random 
import pickle as pc

kol = 10

populacija= []
for i in range (101):
    populacija.append (Model([10,10,10,10,10,9], 33600))
sort=list(reversed(populacija))
b=slice(10)
populacija = (sort[b])
for i in range (91):
    x1 =  Model([kol,kol,kol,kol,kol,9], 33600)
    x2 = Model([kol,kol,kol,kol,kol,9], 33600)
    r1 = x1.eval_model()
    r2 = x2.eval_model()
    dijete = Model([kol,kol,kol,kol,kol,9], 33600)
    dijete.cross(x1,x2)
    populacija=populacija + dijete 









