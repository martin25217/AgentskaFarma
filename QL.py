"""
Q-Learning Training and Demo Script for MountainCar-v0

TRAIN:
    python3 QL.py --train --episodes 50000

DEMO + GIF:
    python3 QL.py --demo --model MountainCar_Qtable.npy --gif mountaincar.gif
"""

import argparse
import os
import time

import gymnasium as gym
import numpy as np

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


# ---------------------------------------------------------
# Seed
# ---------------------------------------------------------

def set_seed(seed):
    np.random.seed(seed)



# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

def make_env(render_mode=None):

    return gym.make(
        "MountainCar-v0",
        render_mode=render_mode
    )



# ---------------------------------------------------------
# Discretization
# ---------------------------------------------------------

def discretize(state, bins):

    low = np.array([
        -1.2,
        -0.07
    ])

    high = np.array([
        0.6,
        0.07
    ])

    ratios = (state - low) / (high - low)

    state = (
        ratios * bins
    ).astype(int)

    state = np.clip(
        state,
        0,
        bins - 1
    )

    return tuple(state)



# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train_qlearning(episodes, seed):

    set_seed(seed)

    env = make_env()

    bins = 20

    q_table = np.zeros(
        (
            bins,
            bins,
            env.action_space.n
        )
    )


    learning_rate = 0.1
    gamma = 0.99

    epsilon = 1.0
    epsilon_decay = 0.99995
    epsilon_min = 0.01


    print("Training MountainCar")
    print("Episodes:", episodes)


    for episode in range(episodes):

        state, _ = env.reset()

        state = discretize(
            state,
            bins
        )

        done = False


        while not done:

            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(
                    q_table[state]
                )


            next_state, reward, terminated, truncated, _ = env.step(action)

            next_state = discretize(
                next_state,
                bins
            )


            old_value = q_table[
                state + (action,)
            ]

            next_max = np.max(
                q_table[next_state]
            )


            q_table[
                state + (action,)
            ] = old_value + learning_rate * (
                reward +
                gamma * next_max -
                old_value
            )


            state = next_state

            done = terminated or truncated


        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay
        )


        if episode % 1000 == 0:
            print(
                f"Episode {episode}, epsilon={epsilon:.3f}"
            )


    filename = "MountainCar_Qtable.npy"

    np.save(
        filename,
        q_table
    )

    print("\nSaved model:")
    print(filename)


    env.close()



# ---------------------------------------------------------
# Demo + GIF recording
# ---------------------------------------------------------

def run_demo(model_path, gif_path):

    if not os.path.exists(model_path):

        print("MODEL NOT FOUND:")
        print(model_path)
        return


    q_table = np.load(model_path)


    env = make_env(
        render_mode="rgb_array"
    )


    bins = 20

    frames = []


    state, _ = env.reset()

    state = discretize(
        state,
        bins
    )


    done = False
    total_reward = 0


    while not done:

        frame = env.render()

        frames.append(frame)


        action = np.argmax(
            q_table[state]
        )


        next_state, reward, terminated, truncated, _ = env.step(action)


        state = discretize(
            next_state,
            bins
        )


        total_reward += reward


        done = terminated or truncated


    frame = env.render()
    frames.append(frame)


    print(
        "Episode reward:",
        total_reward
    )


    env.close()


    print("Creating GIF...")


    clip = ImageSequenceClip(
        frames,
        fps=30
    )


    clip.write_gif(
        gif_path,
        fps=30
    )


    print("GIF saved:")
    print(gif_path)



# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--train",
        action="store_true"
    )


    parser.add_argument(
        "--demo",
        action="store_true"
    )


    parser.add_argument(
        "--episodes",
        type=int,
        default=50000
    )


    parser.add_argument(
        "--model",
        default="MountainCar_Qtable.npy"
    )


    parser.add_argument(
        "--gif",
        default="MountainCar_demo.gif"
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=1
    )


    args = parser.parse_args()


    if args.train:

        train_qlearning(
            args.episodes,
            args.seed
        )


    elif args.demo:

        run_demo(
            args.model,
            args.gif
        )


    else:

        print(
            "Use --train or --demo"
        )