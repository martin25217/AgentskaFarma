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
)#If CUDA available -> network weights → GPU memory; If CUDA not -> network weights → CPU memory
#(Compute Unified Device Architecture) allows programs to use an NVIDIA GPU for general-purpose computing, not just graphics.

print("Environment created")
env = gym.make(
    "ALE/MsPacman-v5"
)


env = AtariPreprocess(env)


env = FrameStackObservation(
    env,
    stack_size=4
)


actions = env.action_space.n#what actions the agent can take - action_space; For a Discrete (fixed number) action space, .n gives the number of possible actions.

print("Agent created")
agent = Agent(
    actions,
    device
)

#______________________________________________________________________________________________
episodes = 10000
batch_size = 32


for episode in range(episodes):

    state, info = env.reset()

    total_reward = 0


    while True:

        action = agent.select_action(state)


        next_state, reward, terminated, truncated, info = env.step(action)


        done = terminated or truncated

#stores one experience in the agent's replay memory
        agent.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )


        agent.train(batch_size)#model.train() (PyTorch built-in); batch - "When training, take 32 examples at a time."


        state = next_state

        total_reward += reward


        if done:
            break

#________________________________________________________________________________________________________________________________________________
    if episode % 10 == 0:
        agent.update_target()


    print(
        episode,
        total_reward,
        agent.epsilon#defined in agent(53)
    )


env.close()