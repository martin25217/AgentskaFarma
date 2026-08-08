"""
DQN Training and Demo Script for MsPacman-v5
--------------------------------------------
Train or test a Deep Q-Learning agent using Stable Baselines3 (SB3).

TRAIN:
    python3 DQL_fixed.py --train --lr 1e-4 --timesteps 1000000

DEMO:
    python3 DQL_fixed.py --demo --model MODEL_PATH.zip
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
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3.common.logger import configure

ENV_ID = "ALE/MsPacman-v5"
N_STACK = 4  # classic Nature-DQN frame stacking

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
def make_train_env(seed, n_envs=1, render=False):
    """
    make_atari_env wraps each sub-env in SB3's AtariWrapper, which handles:
      - NoopReset, MaxAndSkip (frame skipping), episodic-life, fire-on-reset
      - Grayscale + resize to 84x84
      - Reward clipping to {-1, 0, 1}
    VecFrameStack then stacks the last N_STACK frames so the agent can see motion.

    IMPORTANT: use frameskip=1 in env_kwargs so AtariWrapper (not the base ALE env)
    controls frame skipping - otherwise frames get skipped twice.
    """
    env_kwargs = {"frameskip": 1}
    if render:
        env_kwargs["render_mode"] = "rgb_array"
    env = make_atari_env(ENV_ID, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs)
    env = VecFrameStack(env, n_stack=N_STACK)
    return env


# ---------------------------------------------------------
# Create DQN
# ---------------------------------------------------------
def create_model(env, lr, log_dir):
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=lr,
        buffer_size=100_000,       # bigger buffer now that frames are small (84x84x4 uint8)
        learning_starts=10_000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        optimize_memory_usage=False,
        verbose=1,
        tensorboard_log=log_dir,
    )
    logger = configure(log_dir, ["tensorboard", "stdout"])
    model.set_logger(logger)
    return model


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
def train_dqn(lr, timesteps, seed):
    set_seed(seed)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"DQN/logs/DQN_lr_{lr}_seed_{seed}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    env = make_train_env(seed)

    print("\nTraining MsPacman")
    print("Learning rate:", lr)
    print("Seed:", seed)
    print("Steps:", timesteps)
    print("Logs:", log_dir)

    model = create_model(env, lr, log_dir)
    model.learn(total_timesteps=timesteps, progress_bar=True)

    filename = f"DQN_MsPacman_lr-{lr}_seed-{seed}_{timestamp}_{timesteps}steps.zip"
    path = os.path.join(log_dir, filename)
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

    # Same preprocessing pipeline used in training (must match, or predict() breaks)
    env = make_train_env(seed=0, n_envs=1, render=True)

    # VecVideoRecorder is the SB3-native way to record video from a VecEnv
    # (gymnasium's RecordVideo wrapper doesn't compose cleanly with VecFrameStack).
    video_length = 3000  # cap so a stuck agent can't record forever
    env = VecVideoRecorder(
        env,
        video_folder="videos",
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix="MsPacman_DQN",
    )

    model = DQN.load(model_path, device="cpu")
    print("MODEL OK")

    obs = env.reset()
    for ep in range(episodes):
        done = False
        reward_total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            reward_total += reward[0]
            done = bool(dones[0])
        print(f"Episode {ep + 1} reward:", reward_total)

    env.close()
    print("\nVIDEO SAVED IN:")
    print("videos/")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model", type=str)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    if args.train:
        train_dqn(args.lr, args.timesteps, args.seed)
    elif args.demo:
        run_demo(args.model, args.episodes)
    else:
        print("Use --train or --demo")
