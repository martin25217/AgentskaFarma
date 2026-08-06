import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, actions):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(4,32,8,4),
            nn.ReLU(),

            nn.Conv2d(32,64,4,2),
            nn.ReLU(),

            nn.Conv2d(64,64,3,1),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(3136,512),
            nn.ReLU(),

            nn.Linear(512,actions)
        )


    def forward(self,x):

        x = x / 255.0

        return self.network(x)