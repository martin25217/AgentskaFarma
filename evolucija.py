
from gym import ACTION_COUNT, Model
import pickle as pc

kol = 10

x1 =  Model([kol,kol,kol,kol,kol,ACTION_COUNT])
x2 = Model([kol,kol,kol,kol,kol,ACTION_COUNT])
r1 = x1.eval_model()
r2 = x2.eval_model()
print(x1.weights[0][0][0], '\n')
print(x2.weights[0][0][0], '\n')


print(r1, " ", r2, '\n')


dijete = Model([kol,kol,kol,kol,kol,ACTION_COUNT])
dijete.cross(x1,x2,0.5)

print(dijete.eval_model())

