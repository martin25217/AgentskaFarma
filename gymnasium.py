import sys

sys.path.pop(0)

import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(obs.shape, reward, terminated, truncated)
env.close()
