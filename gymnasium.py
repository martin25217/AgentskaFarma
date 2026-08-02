import sys

sys.path.pop(0)

import ale_py
import gymnasium as gym

gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5", obs_type="grayscale", render_mode="rgb_array")
env = gym.wrappers.FlattenObservation(env)

def model(_obs):
    return env.action_space.sample()


obs, info = env.reset()
while True:
    action = model(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs.shape, action, reward)
    if terminated or truncated:
        break
env.close()




