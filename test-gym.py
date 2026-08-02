from gym import Model
import pickle as pc



x1 =  Model([10,10,10,10,10,9], 33600)
x2 = Model([10,10,10,10,10,9], 33600)
r1 = x1.eval_model()
r2 = x2.eval_model()

print(r1, " ", r2, '\n')


dijete = Model([10,10,10,10,10,9], 33600)
dijete.cross(x1,x2)

print(dijete.eval_model())



# pc.dump(x, open('weights.pkl', 'wb'))



