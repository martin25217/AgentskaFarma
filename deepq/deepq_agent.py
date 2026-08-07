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
            dtype=torch.float32,#floating-point numbers, 32 bits per number (fast, memory efficient, supported efficiently on GPUs.)
            device=self.device
        ).unsqueeze(0)#"This is a batch containing exactly one state."; training - multiple states / playing - one state


        with torch.no_grad():#training - calculate how each weight should change -> gameply "Don't build the computation graph. I'm only doing inference (prediction)."

            q_values = self.policy_net(state)


        # Choose action with highest Q value
        return q_values.argmax(1).item()#.argmax(1) - dimensions - numbered, "For each row, find..."; .item() - extracts the value from a tensor that contains exactly one element



    def train(self, batch_size):

        # Wait until memory has enough data
        if len(self.memory) < batch_size:
            return#random.sample(self.memory, 64) -> error (can't sample 64 unique items from a list of only 12); return - stp function


        batch = self.memory.sample(
            batch_size
        )


        states, actions, rewards, next_states, dones = zip(*batch)# * "Give zip each element inside the list as a separate argument"; zip - separate by category
        #Experience 1:
#(state1, action1, reward1, next_state1, done1)
#Experience 2:
#(state2, action2, reward2, next_state2, done2)... ->
#(
#(state1, state2, state3),
#(action1, action2, action3),...
#)

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
            device=self.device#tells PyTorch where to store the tensor (CPU or GPU)
        )


        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )


        dones = torch.tensor(#dones = (False, False, True, False); agent learns that after a terminal state, there is no future value
            dones,
            dtype=torch.float32,
            device=self.device
        )



        # Q value of chosen actions
        current_q = self.policy_net(states)

        current_q = current_q.gather(#gather - creates new tensor by selecting specific values from an input tensor based on the indices provided
            1,#row=0; column=1
            actions.unsqueeze(1)
        ).squeeze(1)
#"From the network's Q-value predictions, select the Q-value corresponding to the action taken in each experience, then remove the extra dimension."


        # Calculate target Q values
        with torch.no_grad():#"Do not calculate or store gradients for the operations inside this block."

            next_q = self.target_net(next_states)#used only to calculate the target Q-value ↑

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

        loss.backward()#backward - calculates gradients for the nn's parameters; "Figure out how much each weight in the nn contributed to the error, calculate the direction it should change."

        self.optimizer.step()#updates the nn's weights



        # Reduce randomness over time
        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay



    def update_target(self):

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )
