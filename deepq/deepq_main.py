print("MAIN FILE RUNNING")
import gymnasium as gym
import ale_py
import torch
print("Program started")

from gymnasium.wrappers import FrameStackObservation

from deepq.deepq_utilis import AtariPreprocess
from deepq.deepq_agent import Agent


gym.register_envs(ale_py)


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Environment created")
env = gym.make(
    "ALE/MsPacman-v5"
)


env = AtariPreprocess(env)


env = FrameStackObservation(
    env,
    stack_size=4
)


actions = env.action_space.n

print("Agent created")
agent = Agent(
    actions,
    device
)


episodes = 10000
batch_size = 32


for episode in range(episodes):

    state, info = env.reset()

    total_reward = 0


    while True:

        action = agent.select_action(state)


        next_state, reward, terminated, truncated, info = env.step(action)


        done = terminated or truncated


        agent.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )


        agent.train(batch_size)


        state = next_state

        total_reward += reward


        if done:
            break


    if episode % 10 == 0:
        agent.update_target()


    print(
        episode,
        total_reward,
        agent.epsilon
    )


env.close()