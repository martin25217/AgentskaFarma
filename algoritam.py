from gym import Model  
import random 
import pickle as pc

kol = 10
par = 30

populacija= []
for i in range (par):
    x = Model([kol,kol,kol,kol,kol,9], 33600)
    r1 = x.eval_model()
    populacija.append([r1,x])

populacija.sort(key=lambda item: item[0], reverse=True)
b = []
for i in range (10):
   b.append(populacija[i])

print(len(populacija))

for j in range(100):
    for i in range (par-10):
        p1=random.choice(populacija)
        
        p2=random.choice(populacija)
        dijete = Model([kol,kol,kol,kol,kol,9], 33600)
        dijete.cross(p1[1],p2[1])
        rew = dijete.eval_model()

        populacija.append([rew, dijete])

    
    populacija.sort(key=lambda item: item[0], reverse=True)
    
    for i in range (10):
       populacija.append(b[i])
    print(len(populacija))
    populacija = populacija[:par]
   # print(len(populacija))
    populacija.sort(key=lambda item: item[0], reverse=True)
    b = []
    for i in range (10):  
        b.append(populacija[i])

    print(len(populacija), '\n' , populacija[0][0])
