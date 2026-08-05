import argparse
import copy
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ale_py
import numpy as np
import pygame
import torch
from pettingzoo.atari import pong_v3
from torch import nn
from torch.distributions import Categorical


PLAYERS = ("first_0", "second_0")
ACTIONS = np.array((1, 4, 5))  # fire/stay, fire/up, fire/down
FEATURES = 8
CHECKPOINT_VERSION = 1


def player_features(frames, player):
    """Small player-relative state; no scripted intercept is exposed to the model."""
    previous, state = frames
    ball_x, ball_y = int(state[49]), int(state[54])
    dx, dy = ball_x - int(previous[49]), ball_y - int(previous[54])
    dx = dx if abs(dx) <= 4 else 0
    dy = dy if abs(dy) <= 4 else 0
    direction = 1 if player == 0 else -1
    own_y, opponent_y = int(state[51 - player]), int(state[50 + player])
    return np.array(
        (
            direction * (ball_x - 128) / 86,
            (ball_y - 124) / 82,
            direction * dx / 4,
            dy / 4,
            (own_y - 124) / 82,
            (opponent_y - 124) / 82,
            (ball_y - own_y) / 165,
            (ball_y - opponent_y) / 165,
        ),
        dtype=np.float32,
    )


def teacher_actions(frames):
    """Training-only labels for the imitation warm start."""
    previous, state = frames
    ball_x, ball_y = int(state[49]), int(state[54])
    dx, dy = ball_x - int(previous[49]), ball_y - int(previous[54])
    actions = np.zeros(2, dtype=np.int64)
    if not (ball_x > 49 and ball_y and 0 < abs(dx) <= 4 and abs(dy) <= 4):
        return actions
    player = 0 if dx > 0 else 1
    target_x, offset = ((188, 6), (66, 4))[player]
    reflected = (ball_y + dy * (target_x - ball_x) / dx - 42) % 328
    target_y = 42 + (reflected if reflected <= 164 else 328 - reflected) - offset
    error = target_y - int(state[51 - player])
    actions[player] = 1 if error < -3 else 2 if error > 3 else 0
    return actions


class Match:
    def __init__(self, sticky=0.0, render_mode=None):
        self.env = pong_v3.parallel_env(
            num_players=2,
            obs_type="ram",
            full_action_space=False,
            max_cycles=100_000,
            auto_rom_install_path=Path(ale_py.__file__).parent,
            render_mode=render_mode,
        )
        self.env.unwrapped.ale.setFloat(b"repeat_action_probability", sticky)
        self.frames = None

    def reset(self, seed=None):
        observations, _ = self.env.reset(seed=seed)
        state = observations[PLAYERS[0]]
        self.frames = np.repeat(state[None], 2, axis=0)
        return self.frames.copy()

    def step(self, action_indices):
        observations, rewards, terminated, truncated, _ = self.env.step(
            dict(zip(PLAYERS, ACTIONS[np.asarray(action_indices)]))
        )
        done = any(terminated.values()) or any(truncated.values())
        reward = np.array([rewards[player] for player in PLAYERS], dtype=np.float32)
        if done:
            return self.reset(), reward, True
        self.frames[0] = self.frames[1]
        self.frames[1] = observations[PLAYERS[0]]
        return self.frames.copy(), reward, False

    def close(self):
        self.env.close()


class Matches:
    def __init__(self, count, workers, sticky):
        self.matches = [Match(sticky) for _ in range(count)]
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def reset(self, seed):
        futures = [self.pool.submit(match.reset, seed + index) for index, match in enumerate(self.matches)]
        return np.stack([future.result() for future in futures])

    def step(self, actions):
        futures = [
            self.pool.submit(match.step, action)
            for match, action in zip(self.matches, actions)
        ]
        results = [future.result() for future in futures]
        return (
            np.stack([result[0] for result in results]),
            np.stack([result[1] for result in results]),
            np.array([result[2] for result in results], dtype=np.float32),
        )

    def close(self):
        self.pool.shutdown()
        for match in self.matches:
            match.close()


def init_layer(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Policy(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.width = width
        self.trunk = nn.Sequential(
            init_layer(nn.Linear(FEATURES, width)),
            nn.Tanh(),
            init_layer(nn.Linear(width, width)),
            nn.Tanh(),
        )
        self.actor = init_layer(nn.Linear(width, len(ACTIONS)), 0.01)
        self.critic = init_layer(nn.Linear(width, 1), 1.0)

    def forward(self, observations):
        hidden = self.trunk(observations)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    @torch.inference_mode()
    def act(self, observations, deterministic=False):
        logits, value = self(observations)
        distribution = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), value


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    return torch.device(name)


def save_checkpoint(path, policy, optimizer=None, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "width": policy.width,
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        **metadata,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("not a self-play Pong checkpoint")
    policy = Policy(checkpoint["width"]).to(device)
    policy.load_state_dict(checkpoint["policy"])
    return policy, checkpoint


def warm_start(policy, device, rounds, steps, seed):
    """DAgger: learn teacher actions on states increasingly visited by the model."""
    rng = np.random.default_rng(seed)
    match = Match()
    frames = match.reset(seed)
    observations, labels = [], []
    parameters = [*policy.trunk.parameters(), *policy.actor.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    try:
        for round_index in range(rounds):
            for _ in range(steps):
                label = teacher_actions(frames)
                observations.extend(player_features(frames, player) for player in range(2))
                labels.extend(label)
                if round_index == 0:
                    actions = label.copy()
                else:
                    batch = torch.as_tensor(
                        np.stack([player_features(frames, player) for player in range(2)]),
                        device=device,
                    )
                    actions = policy.act(batch, deterministic=True)[0].cpu().numpy().copy()
                noise = rng.random(2) < 0.1
                actions[noise] = rng.integers(len(ACTIONS), size=noise.sum())
                frames, _, _ = match.step(actions)

            inputs = torch.as_tensor(np.asarray(observations), device=device)
            targets = torch.as_tensor(np.asarray(labels), device=device)
            for _ in range(6):
                order = torch.randperm(len(inputs), device=device)
                for start in range(0, len(inputs), 2_048):
                    indices = order[start : start + 2_048]
                    logits, _ = policy(inputs[indices])
                    loss = nn.functional.cross_entropy(logits, targets[indices])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            with torch.inference_mode():
                accuracy = (policy(inputs)[0].argmax(1) == targets).float().mean().item()
            print(f"warm_start={round_index + 1}/{rounds} samples={len(inputs)} accuracy={accuracy:.3f}", flush=True)
    finally:
        match.close()


def train(args):
    if not (
        args.hours > 0
        and args.envs >= 2
        and args.rollout >= 2
        and args.workers >= 1
        and args.width >= 1
        and args.warm_rounds >= 0
        and args.warm_steps >= 1
        and args.learning_rate > 0
        and args.epochs >= 1
        and args.minibatch >= 2
        and 0 < args.gamma <= 1
        and 0 <= args.gae_lambda <= 1
        and 0 < args.clip < 1
        and args.entropy >= 0
        and args.imitation >= 0
        and args.value_coefficient >= 0
        and 0 <= args.sticky <= 1
        and args.opponent_updates >= 1
        and 0 <= args.teacher_fraction <= 1
    ):
        raise SystemExit("invalid training parameter")
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    deadline = time.monotonic() + args.hours * 3_600

    if args.resume:
        policy, checkpoint = load_checkpoint(args.resume, device)
        updates = int(checkpoint.get("updates", 0))
        total_frames = int(checkpoint.get("frames", 0))
    else:
        policy, checkpoint, updates, total_frames = Policy(args.width).to(device), {}, 0, 0
        warm_start(policy, device, args.warm_rounds, args.warm_steps, args.seed)

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)
    if args.resume and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    opponent = copy.deepcopy(policy).eval()
    learner_sides = np.arange(args.envs) % 2
    teacher_opponents = np.arange(args.envs) < round(args.envs * args.teacher_fraction)
    matches = Matches(args.envs, min(args.workers, args.envs), args.sticky)
    frames = matches.reset(args.seed)
    started = time.monotonic()
    start_updates = updates

    print(
        f"device={device} envs={args.envs} rollout={args.rollout} "
        f"batch={args.envs * args.rollout}",
        flush=True,
    )
    try:
        while time.monotonic() < deadline:
            obs_buffer = torch.empty((args.rollout, args.envs, FEATURES), device=device)
            action_buffer = torch.empty((args.rollout, args.envs), dtype=torch.long, device=device)
            logprob_buffer = torch.empty_like(action_buffer, dtype=torch.float32)
            reward_buffer = torch.empty_like(logprob_buffer)
            done_buffer = torch.empty_like(logprob_buffer)
            value_buffer = torch.empty_like(logprob_buffer)
            teacher_buffer = torch.empty_like(action_buffer)
            points_for = points_against = 0

            for step in range(args.rollout):
                learner_observations = torch.as_tensor(
                    np.stack(
                        [player_features(frame, side) for frame, side in zip(frames, learner_sides)]
                    ),
                    device=device,
                )
                opponent_observations = torch.as_tensor(
                    np.stack(
                        [player_features(frame, 1 - side) for frame, side in zip(frames, learner_sides)]
                    ),
                    device=device,
                )
                teacher_labels = np.array(
                    [teacher_actions(frame)[side] for frame, side in zip(frames, learner_sides)]
                )
                with torch.inference_mode():
                    actions, logprobs, values = policy.act(learner_observations)
                    opponent_actions = opponent.act(opponent_observations)[0]
                joint_actions = np.empty((args.envs, 2), dtype=np.int64)
                learner_numpy = actions.cpu().numpy()
                opponent_numpy = opponent_actions.cpu().numpy().copy()
                for index in np.flatnonzero(teacher_opponents):
                    opponent_numpy[index] = teacher_actions(frames[index])[1 - learner_sides[index]]
                joint_actions[np.arange(args.envs), learner_sides] = learner_numpy
                joint_actions[np.arange(args.envs), 1 - learner_sides] = opponent_numpy
                frames, rewards, dones = matches.step(joint_actions)
                learner_rewards = rewards[np.arange(args.envs), learner_sides]

                obs_buffer[step] = learner_observations
                action_buffer[step] = actions
                logprob_buffer[step] = logprobs
                value_buffer[step] = values
                teacher_buffer[step] = torch.as_tensor(teacher_labels, device=device)
                reward_buffer[step] = torch.as_tensor(learner_rewards, device=device)
                done_buffer[step] = torch.as_tensor(dones, device=device)
                points_for += int(np.count_nonzero(learner_rewards > 0))
                points_against += int(np.count_nonzero(learner_rewards < 0))

            with torch.no_grad():
                next_observations = torch.as_tensor(
                    np.stack(
                        [player_features(frame, side) for frame, side in zip(frames, learner_sides)]
                    ),
                    device=device,
                )
                next_value = policy(next_observations)[1]
                advantages = torch.zeros_like(reward_buffer)
                last_advantage = torch.zeros(args.envs, device=device)
                for step in reversed(range(args.rollout)):
                    next_nonterminal = 1 - done_buffer[step]
                    following_value = next_value if step == args.rollout - 1 else value_buffer[step + 1]
                    delta = (
                        reward_buffer[step]
                        + args.gamma * following_value * next_nonterminal
                        - value_buffer[step]
                    )
                    last_advantage = (
                        delta
                        + args.gamma * args.gae_lambda * next_nonterminal * last_advantage
                    )
                    advantages[step] = last_advantage
                returns = advantages + value_buffer

            flat_observations = obs_buffer.flatten(0, 1)
            flat_actions = action_buffer.flatten()
            flat_logprobs = logprob_buffer.flatten()
            flat_advantages = advantages.flatten()
            flat_returns = returns.flatten()
            flat_teacher = teacher_buffer.flatten()
            batch_size = len(flat_actions)
            for _ in range(args.epochs):
                order = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, args.minibatch):
                    indices = order[start : start + args.minibatch]
                    logits, values = policy(flat_observations[indices])
                    distribution = Categorical(logits=logits)
                    new_logprobs = distribution.log_prob(flat_actions[indices])
                    ratio = (new_logprobs - flat_logprobs[indices]).exp()
                    batch_advantages = flat_advantages[indices]
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                        batch_advantages.std(unbiased=False) + 1e-8
                    )
                    policy_loss = torch.maximum(
                        -batch_advantages * ratio,
                        -batch_advantages * ratio.clamp(1 - args.clip, 1 + args.clip),
                    ).mean()
                    value_loss = 0.5 * (values - flat_returns[indices]).square().mean()
                    imitation_loss = nn.functional.cross_entropy(
                        logits, flat_teacher[indices]
                    )
                    loss = (
                        policy_loss
                        + args.value_coefficient * value_loss
                        - args.entropy * distribution.entropy().mean()
                        + args.imitation * imitation_loss
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()

            updates += 1
            total_frames += args.envs * args.rollout
            if updates % args.opponent_updates == 0:
                opponent.load_state_dict(policy.state_dict())
            save_checkpoint(
                args.checkpoint,
                policy,
                optimizer,
                updates=updates,
                frames=total_frames,
            )
            fps = int(
                (args.envs * args.rollout * (updates - start_updates))
                / max(time.monotonic() - started, 1e-6)
            )
            print(
                f"update={updates} frames={total_frames} score={points_for}-{points_against} fps={fps} saved={args.checkpoint}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        save_checkpoint(args.checkpoint, policy, optimizer, updates=updates, frames=total_frames)
        matches.close()


def play(checkpoint, device_name, human=False):
    device = resolve_device(device_name)
    policy, _ = load_checkpoint(checkpoint, device)
    policy.eval()
    match = Match(render_mode="human")
    frames = match.reset()
    clock = pygame.time.Clock()
    pygame.display.set_caption("Pong: human vs learned bot" if human else "Pong: learned self-play")
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            observations = torch.as_tensor(
                np.stack([player_features(frames, player) for player in range(2)]),
                device=device,
            )
            actions = policy.act(observations, deterministic=True)[0].cpu().numpy().copy()
            if human:
                keys = pygame.key.get_pressed()
                actions[1] = 1 if keys[pygame.K_UP] or keys[pygame.K_w] else 2 if keys[pygame.K_DOWN] or keys[pygame.K_s] else 0
            frames, _, _ = match.step(actions)
            clock.tick(60)
    finally:
        match.close()


def check():
    device = torch.device("cpu")
    match = Match()
    try:
        frames = match.reset(seed=0)
        observations = np.stack([player_features(frames, player) for player in range(2)])
        assert observations.shape == (2, FEATURES)
        assert np.all((teacher_actions(frames) >= 0) & (teacher_actions(frames) < len(ACTIONS)))
        policy = Policy(16)
        actions = policy.act(torch.as_tensor(observations), deterministic=True)[0]
        frames, rewards, done = match.step(actions.numpy())
        assert frames.shape == (2, 128) and rewards.shape == (2,) and isinstance(done, bool)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            save_checkpoint(path, policy, frames=1)
            loaded, checkpoint = load_checkpoint(path, device)
            assert checkpoint["frames"] == 1
            assert all(torch.equal(a, b) for a, b in zip(policy.parameters(), loaded.parameters()))
    finally:
        match.close()
    print("OK")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and play learned Pong self-play.")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--hours", type=float, default=1)
    training.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    training.add_argument("--envs", type=int, default=32)
    training.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    training.add_argument("--rollout", type=int, default=256)
    training.add_argument("--width", type=int, default=128)
    training.add_argument("--warm-rounds", type=int, default=6)
    training.add_argument("--warm-steps", type=int, default=10_000)
    training.add_argument("--learning-rate", type=float, default=2.5e-4)
    training.add_argument("--epochs", type=int, default=4)
    training.add_argument("--minibatch", type=int, default=1_024)
    training.add_argument("--gamma", type=float, default=0.99)
    training.add_argument("--gae-lambda", type=float, default=0.95)
    training.add_argument("--clip", type=float, default=0.2)
    training.add_argument("--entropy", type=float, default=0.01)
    training.add_argument("--imitation", type=float, default=0.02)
    training.add_argument("--value-coefficient", type=float, default=0.5)
    training.add_argument("--sticky", type=float, default=0.05)
    training.add_argument("--opponent-updates", type=int, default=10)
    training.add_argument("--teacher-fraction", type=float, default=0.25)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--checkpoint", type=Path, default=Path("weights/pong-selfplay.pt"))
    training.add_argument("--resume", type=Path)

    for name in ("watch", "human"):
        playing = commands.add_parser(name)
        playing.add_argument("checkpoint", nargs="?", type=Path, default=Path("weights/pong-selfplay.pt"))
        playing.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    commands.add_parser("check")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "train":
        train(arguments)
    elif arguments.command == "check":
        check()
    else:
        play(arguments.checkpoint, arguments.device, arguments.command == "human")
