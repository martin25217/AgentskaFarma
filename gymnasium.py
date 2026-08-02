import sys

sys.path.pop(0)

import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5", render_mode="human")
obs, info = env.reset()
while True:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(obs.shape, reward, terminated, truncated)
    if terminated or truncated:
        break
env.close()