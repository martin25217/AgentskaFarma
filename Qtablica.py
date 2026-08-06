import numpy as np
import gymnasium as gym
from moviepy import ImageSequenceClip  # moviepy >= 2.0
# if you're on moviepy 1.x instead, comment out the line above and use:
# from moviepy.editor import ImageSequenceClip

# ---------------- Training ----------------
env = gym.make("Taxi-v4")
n_actions = 6
n_states = 500
Q_table = np.zeros((n_states, n_actions))
learning_rate = 0.8
discount_factor = 0.95
exploration_prob = 0.2
n_episodes = 1000

for episode in range(n_episodes):
    state = env.reset()[0]
    done = False

    while not done:
        if np.random.rand() < exploration_prob:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[state])

        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        best_next_action_value = np.max(Q_table[observation])
        Q_table[state][action] += learning_rate * (
            reward + discount_factor * best_next_action_value * (1 - done) - Q_table[state][action]
        )
        state = observation

env.close()
print(Q_table)

# ---------------- Record demo episodes to mp4 ----------------
play_env = gym.make("Taxi-v4", render_mode="rgb_array")

frames = []
n_demo_episodes = 5

for ep in range(n_demo_episodes):
    state, _ = play_env.reset()
    done = False
    frames.append(play_env.render())  # capture initial state too
    while not done:
        action = np.argmax(Q_table[state])
        state, reward, terminated, truncated, info = play_env.step(action)
        done = terminated or truncated
        frames.append(play_env.render())
    print(f"Episode {ep + 1} finished, reward: {reward}")

play_env.close()

clip = ImageSequenceClip(frames, fps=4)
clip.write_videofile("taxi_demo.mp4")

print("Saved video to taxi_demo.mp4")
