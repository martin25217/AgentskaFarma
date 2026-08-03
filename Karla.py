import numpy as np
import matplotlib.pyplot as plt

#???   n_states =        
n_actions = 9         
#???   goal_state = n_states*n_actions-1
Q_table = {}


def get_q_row(state):
    """
    Automatically checks if a situation is unknown.
    If unknown, it builds a fresh row of zeros for it.
    """
    if state not in Q_table:
        # Agent recognizes it's a new situation! Automatically adds it.
        Q_table[state] = np.zeros(n_actions)
        print(f"[Automation] Discovered completely new state: {state}. Added zero-row.")
        
    return Q_table[state]


learning_rate = 0.8
discount_factor = 0.95
exploration_prob = 0.2
n = 1000

next_state, reward, done, info = env.step(action)

#new_row = np.zeros(4).reshape(1, -1)      Q_table = np.append(Q_table, new_row, axis=0)
