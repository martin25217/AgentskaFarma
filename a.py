import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5", render_mode="human")

import pygame

pygame.init()
pygame.display.set_mode((200, 200))

key_to_action = {
    pygame.K_UP: 1,
    pygame.K_RIGHT: 2,
    pygame.K_LEFT: 3,
    pygame.K_DOWN: 4,
}

obs, info = env.reset()

done = False

action = 0  # NOOP

while not done:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    keys = pygame.key.get_pressed()

    action = 0  # NOOP by default
    for key, mapped_action in key_to_action.items():
        if keys[key]:
            action = mapped_action
            break

    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

env.close()
pygame.quit() 