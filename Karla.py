import numpy as np
import matplotlib.pyplot as plt
import ale_py
import gymnasium as gym

gym.register_envs(ale_py)
env = gym.make("ALE/MsPacman-v5",obs_type="grayscale",render_mode="rgb_array")
env = gym.wrappers.FlattenObservation(env)

n_actions = 9         
Q_table = {}

learning_rate = 0.8
discount_factor = 0.95
exploration_prob = 0.2
n_episodes = 1000


def get_q_row(state):
    
    if state not in Q_table:
        Q_table[state] = np.zeros(n_actions)
        print(f"[Automation] Discovered completely new state: {state}. Added zero-row.")
        
    return Q_table[state]




for episode in range(n_episodes):
    state=env.reset()
    terminated=False
    truncated=False

    while not terminated and not truncated:
        if np.random.rand() < exploration_prob:
            action = np.random.choice(n_actions)
        else:
            action = np.argmax(get_q_row(state))
        observation, reward, terminated, truncated, info = env.step(action)

        current_row = get_q_row(state)
        next_row = get_q_row(observation)
        best_next_action_value = np.max(next_row)
        
        Q_table[state][action] += learning_rate * (reward + discount_factor * best_next_action_value * (1 - done) - Q_table[state][action])

        state = observation

