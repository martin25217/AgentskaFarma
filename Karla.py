import numpy as np
import matplotlib.pyplot as plt
import ale_py
import gymnasium as gym

env=gym.make("Taxi-v4")

n_actions = 6
n_states = 500
Q_table = np.zeros((n_states, n_actions))

learning_rate = 0.8
discount_factor = 0.95
exploration_prob = 0.2
n_episodes = 1000

for episode in range(n_episodes):
    state=env.reset()[0]
    done=False
    
    while not done:
        if np.random.rand() < exploration_prob:
            action = env.action_space.sample()
        else:
            
            action = np.argmax(Q_table[state])
        
        observation, reward, terminated, truncated, info = env.step(action)
        done=terminated or truncated

        current_row = Q_table[state]
        next_row = Q_table[observation]
        best_next_action_value = np.max(next_row)

        Q_table[state][action] += learning_rate * (reward + discount_factor * best_next_action_value * (1 -done) - Q_table[state][action])


        state = observation
        
print(Q_table)
play_env = gym.make("Taxi-v4", render_mode="human")

state, _ = play_env.reset()
done = False

while not done:
    action = np.argmax(Q_table[state])

    state, reward, terminated, truncated, info = play_env.step(action)
    done = terminated or truncated

    import time
    time.sleep(0.3)

play_env.close()