import gymnasium as gym
import ale_py
import pygame

from gymnasium.utils.play import play


gym.register_envs(ale_py)

env = gym.make(
    "ALE/MsPacman-v5",
    render_mode="rgb_array"
)

print(env.unwrapped.get_action_meanings())


keys_to_action = {
    (pygame.K_UP,): 1,
    (pygame.K_RIGHT,): 2,
    (pygame.K_LEFT,): 3,
    (pygame.K_DOWN,): 4,
}


play(
    env,
    keys_to_action=keys_to_action,
    fps=15
)