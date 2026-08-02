from numpy import *

rng = random.default_rng()

class model:
    def __init__(self, p1):
        self.p1 = p1
        weights=[]
        biases=[]
        for i in range(len(p1)):
            temp = []
            for j in range (p1[i]):
                tempdublje = []
                for k in range (p1[]):
                    temp.append(rng.random())
                temp.append(rng.random())
            weights.append(temp)
            biases.append(rng.random())
        
        print(weights, '\n')
        print(biases, '\n')
        print(len(weights[0]))



test = model([1000, 1000, 1000, 1000, 1000, 8])

l1 = [1,2]
l2 = [5,2]

zb = matmul(l1,l2)
print(zb)



#rng.integers(1,1000)

