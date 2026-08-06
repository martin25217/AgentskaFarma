import torch
import torch.nn as nn
import torch.optim as optim#optimization algorithms used to train deep learning models, updates network weights and biases by minimizing a loss function
import random
import numpy as np

from deepq.dqn import DQN
from deepq.deepq_rmemory import ReplayMemory


class Agent:

    def __init__(self, actions, device):

        self.device = device
        self.actions = actions


        # Main network (learns)
        self.policy_net = DQN(actions).to(device)#class DQN(nn.Module): -> DQN(actions) - creates a neural network; self.policy_net - stores the neural network inside the agent


        # Target network (stable target values)
        self.target_net = DQN(actions).to(device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )#copies all the learned weights from policy_net to target_net; nece se kopirat svaki put kad se pozove class.

        self.target_net.eval()#used to make predictions


        # Memory storage
        self.memory = ReplayMemory(
            100000#"This replay buffer can store up to 100,000 experiences."
        )


        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=0.00025#learning rate
        )


        self.loss = nn.SmoothL1Loss()


        # Discount factor
        self.gamma = 0.99


        # Exploration settings
        self.epsilon = 1.0#starting exploration rate.
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999995#how quickly epsilon decreases



    def select_action(self, state):

        # Random action (exploration)
        if random.random() < self.epsilon:
            return random.randrange(self.actions)


        # Convert observation to tensor
        state = torch.tensor(
            np.array(state),
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)


        with torch.no_grad():

            q_values = self.policy_net(state)


        # Choose action with highest Q value
        return q_values.argmax(1).item()



    def train(self, batch_size):

        # Wait until memory has enough data
        if len(self.memory) < batch_size:
            return


        batch = self.memory.sample(
            batch_size
        )


        states, actions, rewards, next_states, dones = zip(*batch)



        states = torch.tensor(
            np.array(states),
            dtype=torch.float32,
            device=self.device
        )


        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32,
            device=self.device
        )


        actions = torch.tensor(
            actions,
            device=self.device
        )


        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )


        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )



        # Q value of chosen actions
        current_q = self.policy_net(states)

        current_q = current_q.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)



        # Calculate target Q values
        with torch.no_grad():

            next_q = self.target_net(next_states)

            max_next_q = next_q.max(1)[0]


            target_q = rewards + (
                self.gamma *
                max_next_q *
                (1 - dones)
            )



        loss = self.loss(
            current_q,
            target_q
        )


        # Update network
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()



        # Reduce randomness over time
        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay



    def update_target(self):

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )
