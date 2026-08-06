import argparse
import tempfile
import time
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch
from torch import nn


ROWS, COLUMNS = 22, 10
TARGETS = range(2, 13)
ACTION_COUNT = 4 * len(TARGETS)
FEATURES = ROWS * COLUMNS + 7
CHECKPOINT_VERSION = 1
LINE_SCORES = (0, 1, 2, 3, 10)

# Exact pieces and rotations from the ALE Tetris ROM's blktab.
SHAPES = (
    (((0, 0), (1, 0), (0, 1), (1, 1)),) * 4,
    (((1, 0), (2, 0), (0, 1), (1, 1)), ((1, 0), (1, 1), (2, 1), (2, 2))) * 2,
    (((0, 0), (1, 0), (1, 1), (2, 1)), ((2, 0), (1, 1), (2, 1), (1, 2))) * 2,
    (
        ((1, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (2, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (1, 2)),
        ((1, 0), (0, 1), (1, 1), (1, 2)),
    ),
    (
        ((0, 1), (1, 1), (2, 1), (2, 2)),
        ((1, 0), (1, 1), (0, 2), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (1, 2)),
    ),
    (
        ((0, 1), (1, 1), (2, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
        ((2, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (2, 2)),
    ),
    (((0, 1), (1, 1), (2, 1), (3, 1)), ((1, 0), (1, 1), (1, 2), (1, 3))) * 2,
)


def decode_action(action):
    return divmod(int(action), len(TARGETS))[0], int(action) % len(TARGETS) + 2


def encode_action(rotation, target):
    return rotation * len(TARGETS) + target - 2


def legal_actions(piece):
    legal = np.zeros(ACTION_COUNT, dtype=bool)
    for rotation, shape in enumerate(SHAPES[piece]):
        low = 3 - min(x for x, _ in shape)
        high = 12 - max(x for x, _ in shape)
        for target in range(low, high + 1):
            legal[encode_action(rotation, target)] = True
    return legal


LEGAL_ACTIONS = np.stack([legal_actions(piece) for piece in range(7)])


def place(board, piece, rotation, target):
    shape = SHAPES[piece][rotation]
    if not LEGAL_ACTIONS[piece, encode_action(rotation, target)]:
        return None, 0

    def fits(base):
        return all(base - y >= 0 and not board[base - y, target + x - 3] for x, y in shape)

    base = ROWS - 1
    if not fits(base):
        return None, 0
    while fits(base - 1):
        base -= 1

    result = board.copy()
    for x, y in shape:
        result[base - y, target + x - 3] = 1
    full = result.all(axis=1)
    lines = int(full.sum())
    if lines:
        result = np.concatenate((result[~full], np.zeros((lines, COLUMNS), dtype=np.uint8)))
    return result, lines


def board_quality(board):
    heights = np.array(
        [np.flatnonzero(board[:, column])[-1] + 1 if board[:, column].any() else 0 for column in range(COLUMNS)]
    )
    holes = sum(np.count_nonzero(board[: heights[column], column] == 0) for column in range(COLUMNS))
    return -0.510066 * heights.sum() - 0.35663 * holes - 0.184483 * np.abs(np.diff(heights)).sum()


def heuristic_values(state):
    board = state[: ROWS * COLUMNS].reshape(ROWS, COLUMNS).astype(np.uint8)
    piece = int(state[-7:].argmax())
    legal = np.flatnonzero(LEGAL_ACTIONS[piece])
    values = np.full(ACTION_COUNT, -np.inf, dtype=np.float32)
    for action in legal:
        rotation, target = decode_action(action)
        after, lines = place(board, piece, rotation, target)
        if after is None:
            continue
        values[action] = board_quality(after) + 0.760666 * LINE_SCORES[lines]
    return values


def teacher_action(state):
    values = heuristic_values(state)
    if np.isfinite(values).any():
        return int(np.argmax(values))
    piece = int(state[-7:].argmax())
    return int(np.flatnonzero(LEGAL_ACTIONS[piece])[0])


class PlacementTetris:
    """One learning step places one piece instead of pressing one joystick frame."""

    def __init__(self, sticky=0.0, render_mode=None):
        gym.register_envs(ale_py)
        self.env = gym.make(
            "ALE/Tetris-v5",
            obs_type="ram",
            frameskip=1,
            repeat_action_probability=sticky,
            render_mode=render_mode,
        )
        self.ram = None

    @staticmethod
    def board(ram):
        left, right = ram[:22].copy(), ram[22:44].copy()
        for row, left_mask, right_mask in zip(ram[92:96], ram[96:100], ram[100:104]):
            if row < ROWS:
                left[row] ^= left_mask
                right[row] ^= right_mask
        return np.array(
            [
                [
                    *(int(left[row]) >> np.arange(5, -1, -1) & 1),
                    *(int(right[row]) >> np.arange(4) & 1),
                ]
                for row in range(ROWS)
            ],
            dtype=np.uint8,
        )

    def state(self):
        piece = int(self.ram[107]) >> 4
        return np.concatenate((self.board(self.ram).ravel(), np.eye(7, dtype=np.float32)[piece]))

    def _step(self, action):
        self.ram, reward, terminated, truncated, info = self.env.step(action)
        return reward, terminated or truncated or self.ram[115] == 255, info

    def reset(self, seed=None):
        self.ram, _ = self.env.reset(seed=seed)
        self._step(1)
        for _ in range(120):
            if self.ram[105] == 21 and np.any(self.ram[92:104]):
                return self.state()
            self._step(0)
        raise RuntimeError("Tetris did not start")

    def step(self, action):
        rotation, target = decode_action(action)
        before = self.board(self.ram)
        desired_rotation = rotation * 4

        def landed(info):
            for _ in range(4):
                self._step(0)
            after = self.board(self.ram)
            lines = max(0, (int(before.sum()) + 4 - int(after.sum())) // COLUMNS)
            return self.state(), lines, False, {**info, "lines": lines}

        dropped = int(self.ram[105]) < 21
        for _ in range(120):
            current_rotation, current_target = int(self.ram[107]) & 15, int(self.ram[106])
            if current_rotation != desired_rotation:
                command = 1
            elif current_target < target:
                command = 2
            elif current_target > target:
                command = 3
            else:
                break
            _, done, info = self._step(command)
            if done:
                return self.state(), 0, True, info
            dropped |= int(self.ram[105]) < 21
            if dropped and self.ram[105] == 21:
                return landed(info)

        for _ in range(300):
            _, done, info = self._step(4)
            if done:
                return self.state(), 0, True, info
            dropped |= int(self.ram[105]) < 21
            if dropped and self.ram[105] == 21:
                return landed(info)
        raise RuntimeError("piece did not land")

    def close(self):
        self.env.close()


class Policy(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.width = width
        self.network = nn.Sequential(
            nn.Linear(FEATURES, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, ACTION_COUNT),
        )

    def forward(self, states):
        logits = self.network(states)
        pieces = states[..., -7:].argmax(-1)
        mask = torch.as_tensor(LEGAL_ACTIONS, device=states.device)[pieces]
        return logits.masked_fill(~mask, -1e9)

    @torch.inference_mode()
    def act(self, states):
        return self(states).argmax(-1)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.states = np.empty((capacity, FEATURES), dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, FEATURES), dtype=np.uint8)
        self.dones = np.empty(capacity, dtype=bool)
        self.size = self.index = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, count, rng, device):
        indices = rng.integers(self.size, size=count)
        return (
            torch.as_tensor(self.states[indices], dtype=torch.float32, device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=device),
            torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device),
        )


def guided_action(policy, state, device, guide):
    with torch.inference_mode():
        q_values = policy(torch.as_tensor(state, device=device).unsqueeze(0))[0].cpu().numpy()
    heuristic = heuristic_values(state)
    legal = np.isfinite(heuristic)
    if not legal.any():
        return int(np.argmax(q_values))
    legal_actions = np.flatnonzero(legal)
    q_values = q_values[legal]
    heuristic = heuristic[legal]
    q_values = (q_values - q_values.mean()) / (q_values.std() + 1e-6)
    heuristic = (heuristic - heuristic.mean()) / (heuristic.std() + 1e-6)
    return int(legal_actions[np.argmax(q_values + guide * heuristic)])


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    return torch.device(name)


def save_checkpoint(path, policy, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {"version": CHECKPOINT_VERSION, "width": policy.width, "policy": policy.state_dict(), **metadata},
        temporary,
    )
    temporary.replace(path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("not a Tetris placement-policy checkpoint")
    policy = Policy(checkpoint["width"]).to(device)
    policy.load_state_dict(checkpoint["policy"])
    return policy, checkpoint


def transition_reward(state, next_state, lines, done, shaping):
    before = board_quality(state[: ROWS * COLUMNS].reshape(ROWS, COLUMNS))
    after = board_quality(next_state[: ROWS * COLUMNS].reshape(ROWS, COLUMNS))
    return 4.0 * LINE_SCORES[lines] + shaping * (after - before) - 5.0 * done


def evaluate(policy, device, guide, episodes, max_pieces):
    env = PlacementTetris()
    scores = []
    try:
        for episode in range(episodes):
            state = env.reset(10_000 + episode)
            score = 0
            for _ in range(max_pieces):
                state, reward, done, _ = env.step(guided_action(policy, state, device, guide))
                score += LINE_SCORES[reward]
                if done:
                    break
            scores.append(score)
    finally:
        env.close()
    return float(np.mean(scores))


def train(args):
    if not (
        args.steps >= 0
        and args.hours > 0
        and args.warmup >= args.batch
        and args.batch >= 2
        and args.replay >= args.warmup
        and args.width >= 1
        and args.learning_rate > 0
        and 0 < args.gamma <= 1
        and 0 <= args.exploration <= 1
        and 0 <= args.teacher_fraction <= 1
        and 0 <= args.final_epsilon <= args.initial_epsilon <= 1
        and args.epsilon_decay >= 1
        and args.target_update >= 1
        and args.train_every >= 1
        and args.report_every >= 1
        and args.imitation_epochs >= 0
        and args.imitation >= 0
        and args.shaping >= 0
        and np.isfinite(args.guide)
        and args.guide >= 0
        and args.eval_episodes >= 1
        and args.eval_pieces >= 1
        and 0 <= args.sticky <= 1
    ):
        raise SystemExit("training parameters must be positive")
    device = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    policy = Policy(args.width).to(device)
    target = Policy(args.width).to(device).eval()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    replay = ReplayBuffer(args.replay)
    demonstrations, labels = [], []
    env = PlacementTetris(args.sticky)
    state = env.reset(args.seed)
    episodes = warmup_lines = warmup_tetrises = 0
    try:
        for _ in range(args.warmup):
            label = teacher_action(state)
            demonstrations.append(state)
            labels.append(label)
            piece = int(state[-7:].argmax())
            action = (
                int(rng.choice(np.flatnonzero(LEGAL_ACTIONS[piece])))
                if rng.random() < args.exploration
                else label
            )
            next_state, cleared, done, _ = env.step(action)
            replay.add(
                state,
                action,
                transition_reward(state, next_state, cleared, done, args.shaping),
                next_state,
                done,
            )
            warmup_lines += cleared
            warmup_tetrises += cleared == 4
            state = env.reset(args.seed + episodes + 1) if done else next_state
            episodes += done

        inputs = torch.as_tensor(np.asarray(demonstrations), device=device)
        targets = torch.as_tensor(labels, device=device)
        for _ in range(args.imitation_epochs):
            for batch in torch.randperm(len(inputs), device=device).split(args.batch):
                loss = nn.functional.cross_entropy(policy(inputs[batch]), targets[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        target.load_state_dict(policy.state_dict())

        print(
            f"warmup={args.warmup} lines={warmup_lines} tetrises={warmup_tetrises} deaths={episodes} "
            f"guide={args.guide:g} device={device}",
            flush=True,
        )
        started = time.monotonic()
        deadline = started + args.hours * 3_600
        best_score = float("-inf")
        if args.checkpoint.is_file():
            incumbent, _ = load_checkpoint(args.checkpoint, device)
            incumbent.eval()
            best_score = evaluate(
                incumbent, device, args.guide, args.eval_episodes, args.eval_pieces
            )
            print(f"incumbent={args.checkpoint} eval={best_score:g}", flush=True)
        online_lines = online_tetrises = online_episodes = step = 0
        last_loss = float("nan")
        while time.monotonic() < deadline and (args.steps == 0 or step < args.steps):
            step += 1
            epsilon = args.initial_epsilon + min(step / args.epsilon_decay, 1) * (
                args.final_epsilon - args.initial_epsilon
            )
            piece = int(state[-7:].argmax())
            if rng.random() < args.teacher_fraction:
                action = teacher_action(state)
            elif rng.random() < epsilon:
                action = int(rng.choice(np.flatnonzero(LEGAL_ACTIONS[piece])))
            else:
                action = guided_action(policy, state, device, args.guide)
            next_state, cleared, done, _ = env.step(action)
            replay.add(
                state,
                action,
                transition_reward(state, next_state, cleared, done, args.shaping),
                next_state,
                done,
            )
            online_lines += cleared
            online_tetrises += cleared == 4
            state = env.reset(args.seed + episodes + 1) if done else next_state
            episodes += done
            online_episodes += done

            if step % args.train_every == 0:
                batch_states, batch_actions, rewards, next_states, dones = replay.sample(
                    args.batch, rng, device
                )
                with torch.no_grad():
                    next_actions = policy(next_states).argmax(1, keepdim=True)
                    next_values = target(next_states).gather(1, next_actions).squeeze(1)
                    expected = rewards + args.gamma * (1 - dones) * next_values
                values = policy(batch_states).gather(1, batch_actions[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(values, expected)
                if args.imitation:
                    demonstration_batch = torch.randint(
                        len(inputs), (args.batch,), device=device
                    )
                    loss = loss + args.imitation * nn.functional.cross_entropy(
                        policy(inputs[demonstration_batch]), targets[demonstration_batch]
                    )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 10)
                optimizer.step()
                last_loss = loss.item()

            if step % args.target_update == 0:
                target.load_state_dict(policy.state_dict())
            finished = args.steps and step == args.steps
            if step % args.report_every == 0 or finished:
                score = evaluate(
                    policy, device, args.guide, args.eval_episodes, args.eval_pieces
                )
                saved = score > best_score
                if saved:
                    best_score = score
                    save_checkpoint(
                        args.checkpoint,
                        policy,
                        algorithm="guided-double-dqn",
                        steps=step,
                        samples=args.warmup + step,
                        guide=args.guide,
                        eval_points=best_score,
                    )
                elapsed = max(time.monotonic() - started, 1e-6)
                saved_text = f" saved={args.checkpoint}" if saved else ""
                print(
                    f"step={step} epsilon={epsilon:.3f} loss={last_loss:.4f} "
                    f"train_lines={online_lines} tetrises={online_tetrises} deaths={online_episodes} "
                    f"eval_points={score:g} "
                    f"best_points={best_score:g} pps={step / elapsed:.0f}{saved_text}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        env.close()


def play(checkpoint, device_name, use_teacher, guide=None):
    device = resolve_device(device_name)
    if use_teacher:
        policy = None
    else:
        policy, metadata = load_checkpoint(checkpoint, device)
        policy.eval()
        guide = metadata.get("guide", 16.0) if guide is None else guide
        if not np.isfinite(guide) or guide < 0:
            raise SystemExit("guide must be finite and non-negative")
    env = PlacementTetris(render_mode="human")
    state = env.reset()
    lines = score = 0
    print(f"Ctrl-C to stop" + ("" if use_teacher else f"; guide={guide:g}"), flush=True)
    try:
        while True:
            action = (
                teacher_action(state)
                if use_teacher
                else guided_action(policy, state, device, guide)
            )
            state, reward, done, _ = env.step(action)
            lines += reward
            score += LINE_SCORES[reward]
            if reward:
                print(
                    f"lines={lines} score={score}" + (" TETRIS!" if reward == 4 else ""),
                    flush=True,
                )
            if done:
                print(f"game_over lines={lines} score={score}", flush=True)
                state, lines, score = env.reset(), 0, 0
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def check():
    assert LINE_SCORES == (0, 1, 2, 3, 10)
    board = np.zeros((ROWS, COLUMNS), dtype=np.uint8)
    board[0, :6] = 1
    after, lines = place(board, 6, 0, 9)
    assert lines == 1 and after.sum() == 0
    state = np.concatenate((board.ravel(), np.eye(7, dtype=np.float32)[6]))
    assert LEGAL_ACTIONS[6, teacher_action(state)]
    assert guided_action(Policy(16), state, torch.device("cpu"), 16) in np.flatnonzero(
        LEGAL_ACTIONS[6]
    )
    blocked = np.concatenate(
        (np.ones(ROWS * COLUMNS, dtype=np.float32), np.eye(7, dtype=np.float32)[6])
    )
    assert LEGAL_ACTIONS[6, teacher_action(blocked)]
    assert LEGAL_ACTIONS[6, guided_action(Policy(16), blocked, torch.device("cpu"), 16)]

    env = PlacementTetris()
    try:
        state = env.reset(seed=0)
        expected = teacher_action(state)
        predicted, expected_lines = place(
            state[: ROWS * COLUMNS].reshape(ROWS, COLUMNS).astype(np.uint8),
            int(state[-7:].argmax()),
            *decode_action(expected),
        )
        state, lines, done, _ = env.step(expected)
        assert not done and lines == expected_lines
        assert np.array_equal(state[: ROWS * COLUMNS].reshape(ROWS, COLUMNS), predicted)
    finally:
        env.close()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "policy.pt"
        policy = Policy(16)
        save_checkpoint(path, policy, samples=1)
        loaded, metadata = load_checkpoint(path, torch.device("cpu"))
        assert metadata["samples"] == 1
        assert all(torch.equal(a, b) for a, b in zip(policy.parameters(), loaded.parameters()))
    print("OK")


def parse_args():
    parser = argparse.ArgumentParser(description="Fast placement-level Double DQN for ALE/Tetris-v5.")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--hours", type=float, default=1, help="maximum training time")
    training.add_argument("--steps", type=int, default=0, help="optional placement limit; 0 uses hours")
    training.add_argument("--warmup", type=int, default=2_000)
    training.add_argument("--imitation-epochs", type=int, default=3)
    training.add_argument("--imitation", type=float, default=0.1)
    training.add_argument("--batch", type=int, default=256)
    training.add_argument("--replay", type=int, default=50_000)
    training.add_argument("--width", type=int, default=128)
    training.add_argument("--learning-rate", type=float, default=2.5e-4)
    training.add_argument("--gamma", type=float, default=0.99)
    training.add_argument("--shaping", type=float, default=0.05)
    training.add_argument("--initial-epsilon", type=float, default=0.1)
    training.add_argument("--final-epsilon", type=float, default=0.01)
    training.add_argument("--epsilon-decay", type=int, default=50_000)
    training.add_argument("--target-update", type=int, default=1_000)
    training.add_argument("--train-every", type=int, default=4)
    training.add_argument("--report-every", type=int, default=1_000)
    training.add_argument("--teacher-fraction", type=float, default=0.05)
    training.add_argument("--exploration", type=float, default=0.05)
    training.add_argument("--guide", type=float, default=16.0)
    training.add_argument("--eval-episodes", type=int, default=1)
    training.add_argument("--eval-pieces", type=int, default=1_000)
    training.add_argument("--sticky", type=float, default=0.0)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    training.add_argument("--checkpoint", type=Path, default=Path("weights/tetris-double-dqn.pt"))
    watching = commands.add_parser("watch")
    watching.add_argument("checkpoint", nargs="?", type=Path, default=Path("weights/tetris-double-dqn.pt"))
    watching.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    watching.add_argument("--guide", type=float)
    commands.add_parser("heuristic")
    commands.add_parser("check")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "train":
        train(arguments)
    elif arguments.command == "check":
        check()
    else:
        play(
            getattr(arguments, "checkpoint", Path("weights/tetris-double-dqn.pt")),
            getattr(arguments, "device", "auto"),
            arguments.command == "heuristic",
            getattr(arguments, "guide", None),
        )
