# calibrate.py
import sys
sys.path.pop(0)
import ale_py
import gymnasium as gym
from numpy import unique

gym.register_envs(ale_py)
env = gym.make("ALE/MsPacman-v5", obs_type="rgb", render_mode="rgb_array", repeat_action_probability=0.0)

obs, info = env.reset()

# step a few frames so pellets/sprites are in a normal mid-game state
for _ in range(10):
    obs, reward, terminated, truncated, info = env.step(0)

flat = obs.reshape(-1, 3)
unique_colors, counts = unique(flat, axis=0, return_counts=True)

# sort by frequency — background will dominate, sprites/pellets will be rarer
order = counts.argsort()[::-1]
#for c, n in zip(unique_colors[order], counts[order]):
    #print(c, n)

env.close()