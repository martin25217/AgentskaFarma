
eval_modeli=[]
for model in modeli:
    eval_modeli.append((_,model))
    a=[2,5,6,8,3]


sort=list(reversed(a))
print(sort)
b=slice(10)
print(sort[b])

import random
agenti=[1,4,6,3,9,8,34,64,654,324,544,]
p1=random.choice(agenti)
agenti.pop(agenti.index(p1))
p2=random.choice(agenti)
agenti.pop(agenti.index(p2))
j=[p1,p2]
print(j,agenti)