"""
DQN Training and Demo Script for MsPacman-v5
--------------------------------------------
Train or test a Deep Q-Learning agent using Stable Baselines3 (SB3).

TRAIN:
    python3 DQL --train --lr 0.0001 --timesteps 250000

DEMO:
    python3 DQL --demo --model MODEL_PATH.zip

"""

import argparse
import os
import time

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import numpy as np
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from gymnasium.wrappers import RecordVideo


# ---------------------------------------------------------
# Seed
# ---------------------------------------------------------

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

def make_env(render=False):

    if render:
        env = gym.make(
            "ALE/MsPacman-v5",
            render_mode="rgb_array",
            frameskip=4
        )
    else:
        env = gym.make(
            "ALE/MsPacman-v5",
            frameskip=4
        )

    env = Monitor(env)

    return env



# ---------------------------------------------------------
# Create DQN
# ---------------------------------------------------------

def create_model(env, lr, log_dir):

    model = DQN(
        "CnnPolicy",
        env,

        learning_rate=lr,

        buffer_size=50000,
        learning_starts=10000,

        batch_size=32,

        gamma=0.99,

        train_freq=4,
        gradient_steps=1,

        target_update_interval=10000,

        exploration_fraction=0.1,
        exploration_final_eps=0.01,

        verbose=1,

        tensorboard_log=log_dir
    )


    logger = configure(
        log_dir,
        ["tensorboard"]
    )

    model.set_logger(logger)


    return model



# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train_dqn(lr, timesteps, seed):

    set_seed(seed)

    timestamp = time.strftime("%Y%m%d-%H%M%S")


    log_dir = (
        f"DQN/logs/"
        f"DQN_lr_{lr}_seed_{seed}_{timestamp}"
    )


    os.makedirs(
        log_dir,
        exist_ok=True
    )


    env = make_env()


    print("\nTraining MsPacman")
    print("Learning rate:", lr)
    print("Seed:", seed)
    print("Steps:", timesteps)
    print("Logs:", log_dir)


    model = create_model(
        env,
        lr,
        log_dir
    )


    model.learn(
        total_timesteps=timesteps,
        progress_bar=True
    )


    filename = (
        f"DQN_MsPacman_lr-{lr}"
        f"_seed-{seed}"
        f"_{timestamp}"
        f"_{timesteps}steps.zip"
    )


    path = os.path.join(
        log_dir,
        filename
    )


    model.save(path)


    print("\nMODEL SAVED:")
    print(path)


    env.close()



# ---------------------------------------------------------
# Demo + Video
# ---------------------------------------------------------

def run_demo(model_path, episodes):


    if not os.path.exists(model_path):

        print("MODEL NOT FOUND:")
        print(model_path)

        return



    print("\nLoading model:")
    print(model_path)


    env = gym.make(
        "ALE/MsPacman-v5",
        render_mode="rgb_array",
        frameskip=4
    )


    env = RecordVideo(
        env,
        video_folder="videos",
        name_prefix="MsPacman_DQN"
    )


    print("ENV OK")


    model = DQN.load(
        model_path,
        device="cpu"
    )


    print("MODEL OK")



    for ep in range(episodes):

        obs, info = env.reset()

        terminated = False
        truncated = False

        reward_total = 0


        while not (terminated or truncated):

            action, _ = model.predict(
                obs,
                deterministic=True
            )


            obs, reward, terminated, truncated, info = env.step(action)


            reward_total += reward



        print(
            f"Episode {ep+1} reward:",
            reward_total
        )


    env.close()


    print("\nVIDEO SAVED IN:")
    print("videos/")



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
        "--lr",
        type=float,
        default=0.0001
    )


    parser.add_argument(
        "--timesteps",
        type=int,
        default=5000
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=1
    )


    parser.add_argument(
        "--model",
        type=str
    )


    parser.add_argument(
        "--episodes",
        type=int,
        default=3
    )



    args = parser.parse_args()



    if args.train:

        train_dqn(
            args.lr,
            args.timesteps,
            args.seed
        )


    elif args.demo:


        run_demo(
            args.model,
            args.episodes
        )


    else:

        print(
            "Use --train or --demo"
        )