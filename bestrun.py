from gym import Model  
import random 
import pickle as pc

model = pc.load(open('peakmodel.pkl', 'rb'))
model.eval_model( )
print("score: ", model.eval_model( ))
    
