import torch
import torch.nn as nn


class DQN(nn.Module):#nn.Module - base class for all neural networks in PyTorch

    def __init__(self, actions):#__init__ (initialise); in practice - setting the attributes the methods expect it to have.



        super().__init__()
        #if you have an init method for a Monster class, and you have a child class called Goblin, you can initialize values from the Monster class inside of a Goblin instance
        #iz druge klase?od kud?

        self.network = nn.Sequential(#pamcenje zadnjih 4 stanja???

            nn.Conv2d(4,32,8,4),#2d convulsion (in_chanels, out_chanels, kernel_size, stride)
            nn.ReLU(),#(Rectified Linear Activation Function)ne kužim ??????

            nn.Conv2d(32,64,4,2),
            nn.ReLU(),

            nn.Conv2d(64,64,3,1),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(3136,512),
            nn.ReLU(),

            nn.Linear(512,actions)
        )

#defines how input data flows through the network to produce an output.
    def forward(self,x):

        x = x / 255.0

        return self.network(x)#definirano u __init__