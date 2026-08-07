from collections import deque#(Double-Ended Queue) allows adding and removing elements from both ends of a sequence
import random


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque(
            maxlen=capacity#creates a double-ended queue with a fixed maximum length
        )


    def push(self, state, action, reward, next_state, done):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )


    def sample(self, batch_size):

        return random.sample(
            self.memory,
            batch_size
        )


    def __len__(self):

        return len(self.memory)
