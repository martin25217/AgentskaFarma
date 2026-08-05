from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from pathlib import Path

import ale_py
import numpy as np
import pygame
import torch
import torch.nn.functional as F
from pettingzoo.atari import pong_v3


MUTATION_RATE = 0.1
PLAYERS = ("first_0", "second_0")
PLAYER_ACTION_COUNT = 6
ACTION_COUNT = len(PLAYERS) * PLAYER_ACTION_COUNT
FRAME_STACK = 4
FRAME_SKIP = 4
FRAME_SIZE = 84
ROLLOUT_STEPS = 10_000
CONV_LAYERS = ((4, 8, 8, 4), (8, 16, 4, 2))
DENSE_INPUT = 16 * 9 * 9
RESIZE_ROWS = np.linspace(0, 209, FRAME_SIZE, dtype=int)[:, None]
RESIZE_COLUMNS = np.linspace(0, 159, FRAME_SIZE, dtype=int)


class ContinuousPong:
    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.clock = pygame.time.Clock() if render_mode == "human" else None
        self.env = pong_v3.parallel_env(
            num_players=2,
            obs_type="grayscale_image",
            full_action_space=False,
            max_cycles=100_000,
            auto_rom_install_path=Path(ale_py.__file__).parent,
            render_mode=render_mode,
        )
        self.env.unwrapped.ale.setFloat(b"repeat_action_probability", 0.25)

    def reset(self, **kwargs):
        observations, info = self.env.reset(**kwargs)
        frame = self._frame(observations)
        self.frames = np.repeat(frame[None], FRAME_STACK, axis=0)
        if self.render_mode == "human":
            self.env.render()
            pygame.display.set_caption("Pong cooperative replay")
        return self.frames.copy(), info

    def step(self, actions):
        score = 0
        recent_frames = []
        for _ in range(FRAME_SKIP):
            observations, rewards, terminated, truncated, info = self.env.step(
                dict(zip(PLAYERS, map(int, actions)))
            )
            if self.clock:
                if any(event.type == pygame.QUIT for event in pygame.event.get()):
                    raise SystemExit
                self.clock.tick(60)
            score -= sum(abs(rewards[player]) for player in PLAYERS)
            recent_frames.append(self._frame(observations))
            if any(terminated.values()) or any(truncated.values()):
                observations, info = self.env.reset()
                frame = self._frame(observations)
                self.frames = np.repeat(frame[None], FRAME_STACK, axis=0)
                return self.frames.copy(), score, False, False, info

        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = np.maximum(recent_frames[-2], recent_frames[-1])
        return self.frames.copy(), score, False, False, info

    def _frame(self, observations):
        return observations[PLAYERS[0]][..., 0][RESIZE_ROWS, RESIZE_COLUMNS]

    def close(self):
        self.env.close()


def make_env(render_mode=None):
    return ContinuousPong(render_mode)


def resolve_device(device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Install the CUDA build of PyTorch and check your NVIDIA driver.")
    return torch.device(device)


class Model:
    def __init__(self, sloj, device="auto"):
        self.sloj = list(sloj)
        self.device = resolve_device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.conv_weights = [
            torch.randn(out_channels, in_channels, kernel, kernel, device=self.device, dtype=self.dtype)
            / sqrt(in_channels * kernel * kernel)
            for in_channels, out_channels, kernel, _ in CONV_LAYERS
        ]
        self.conv_biases = [
            torch.zeros(out_channels, device=self.device, dtype=self.dtype)
            for _, out_channels, _, _ in CONV_LAYERS
        ]
        self.weights = [
            torch.randn(fan_in, fan_out, device=self.device, dtype=self.dtype) / sqrt(fan_in)
            for fan_in, fan_out in zip([DENSE_INPUT, *sloj[:-1]], sloj)
        ]
        self.biases = [torch.zeros(size, device=self.device, dtype=self.dtype) for size in sloj]

    @classmethod
    def from_tensors(
        cls, sloj, conv_weights, conv_biases, weights, biases, device="auto", metadata=None
    ):
        model = cls.__new__(cls)
        model.sloj = list(sloj)
        model.device = resolve_device(device)
        model.dtype = torch.float16 if model.device.type == "cuda" else torch.float32
        model.conv_weights = [
            torch.as_tensor(value, dtype=model.dtype, device=model.device) for value in conv_weights
        ]
        model.conv_biases = [
            torch.as_tensor(value, dtype=model.dtype, device=model.device) for value in conv_biases
        ]
        model.weights = [torch.as_tensor(value, dtype=model.dtype, device=model.device) for value in weights]
        model.biases = [torch.as_tensor(value, dtype=model.dtype, device=model.device) for value in biases]
        model.metadata = metadata or {}
        return model

    def cross(self, x1, x2, sigma):
        fan_ins = [in_channels * kernel * kernel for in_channels, _, kernel, _ in CONV_LAYERS]
        self.conv_weights = [
            self._cross_tensor(a, b, sigma / sqrt(fan_in))
            for a, b, fan_in in zip(x1.conv_weights, x2.conv_weights, fan_ins)
        ]
        self.conv_biases = [
            self._cross_tensor(a, b, sigma / sqrt(fan_in))
            for a, b, fan_in in zip(x1.conv_biases, x2.conv_biases, fan_ins)
        ]
        self.weights = [
            self._cross_tensor(a, b, sigma / sqrt(fan_in))
            for a, b, fan_in in zip(x1.weights, x2.weights, [DENSE_INPUT, *self.sloj[:-1]])
        ]
        self.biases = [
            self._cross_tensor(a, b, sigma / sqrt(fan_in))
            for a, b, fan_in in zip(x1.biases, x2.biases, [DENSE_INPUT, *self.sloj[:-1]])
        ]

    def _cross_tensor(self, a, b, step):
        a, b = a.to(self.device), b.to(self.device)
        child = torch.where(torch.rand_like(a) < 0.5, a, b)
        return child + (torch.rand_like(child) < MUTATION_RATE) * torch.randn_like(child) * step

    @torch.inference_mode()
    def feed_forward(self, observation):
        value = torch.as_tensor(observation, dtype=self.dtype, device=self.device).unsqueeze(0)
        value = value / 127.5 - 1
        for weights, biases, (_, _, _, stride) in zip(
            self.conv_weights, self.conv_biases, CONV_LAYERS
        ):
            value = torch.tanh(F.conv2d(value, weights, biases, stride=stride))
        value = value.flatten()
        for weights, biases in zip(self.weights, self.biases):
            value = torch.tanh(value @ weights + biases)
        return value

    def eval_model(self, render_mode=None, seed=None, steps=ROLLOUT_STEPS):
        env = make_env(render_mode)
        try:
            observation, _ = env.reset(seed=seed)
            score = 0.0
            for _ in range(steps):
                actions = (
                    self.feed_forward(observation)
                    .reshape(len(PLAYERS), PLAYER_ACTION_COUNT)
                    .argmax(1)
                    .cpu()
                    .numpy()
                )
                observation, reward, _, _, _ = env.step(actions)
                score += reward
            return score
        finally:
            env.close()

    def save(self, path, **metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "version": 2,
            "sloj": self.sloj,
            "conv_weights": [value.detach().cpu() for value in self.conv_weights],
            "conv_biases": [value.detach().cpu() for value in self.conv_biases],
            "weights": [value.detach().cpu() for value in self.weights],
            "biases": [value.detach().cpu() for value in self.biases],
            **metadata,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="auto"):
        path = Path(path)
        if path.suffix == ".pkl":
            raise ValueError("legacy checkpoints cannot be used with four-frame convolution models")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("version") != 2:
            raise ValueError("checkpoint predates the four-frame convolution model")
        return cls.from_tensors(
            checkpoint["sloj"],
            checkpoint["conv_weights"],
            checkpoint["conv_biases"],
            checkpoint["weights"],
            checkpoint["biases"],
            device,
            {
                key: value
                for key, value in checkpoint.items()
                if key not in {"version", "sloj", "conv_weights", "conv_biases", "weights", "biases"}
            },
        )


class Population:
    def __init__(self, count, sloj, device="auto"):
        self.count = count
        self.sloj = list(sloj)
        self.device = resolve_device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.conv_weights = [
            torch.randn(
                count, out_channels, in_channels, kernel, kernel, device=self.device, dtype=self.dtype
            )
            / sqrt(in_channels * kernel * kernel)
            for in_channels, out_channels, kernel, _ in CONV_LAYERS
        ]
        self.conv_biases = [
            torch.zeros(count, out_channels, device=self.device, dtype=self.dtype)
            for _, out_channels, _, _ in CONV_LAYERS
        ]
        self.weights = [
            torch.randn(count, fan_in, fan_out, device=self.device, dtype=self.dtype) / sqrt(fan_in)
            for fan_in, fan_out in zip([DENSE_INPUT, *sloj[:-1]], sloj)
        ]
        self.biases = [torch.zeros(count, size, device=self.device, dtype=self.dtype) for size in sloj]

    def actions(self, observations):
        value = torch.as_tensor(observations, dtype=self.dtype, device=self.device)
        value = value / 127.5 - 1
        with torch.inference_mode():
            for weights, biases, (_, out_channels, _, stride) in zip(
                self.conv_weights, self.conv_biases, CONV_LAYERS
            ):
                _, channels, height, width = value.shape
                value = F.conv2d(
                    value.reshape(1, self.count * channels, height, width),
                    weights.reshape(self.count * out_channels, channels, *weights.shape[-2:]),
                    biases.flatten(),
                    stride=stride,
                    groups=self.count,
                )
                value = torch.tanh(value.reshape(self.count, out_channels, *value.shape[-2:]))
            value = value.flatten(1)
            for weights, biases in zip(self.weights, self.biases):
                value = torch.tanh(torch.bmm(value.unsqueeze(1), weights).squeeze(1) + biases)
        return (
            value.reshape(self.count, len(PLAYERS), PLAYER_ACTION_COUNT)
            .argmax(2)
            .cpu()
            .numpy()
        )

    def best(self, index):
        return Model.from_tensors(
            self.sloj,
            [value[index] for value in self.conv_weights],
            [value[index] for value in self.conv_biases],
            [value[index] for value in self.weights],
            [value[index] for value in self.biases],
            self.device,
        )

    def inject(self, model):
        for target, source in zip(
            [*self.conv_weights, *self.conv_biases, *self.weights, *self.biases],
            [*model.conv_weights, *model.conv_biases, *model.weights, *model.biases],
        ):
            target[0].copy_(source.to(self.device))

    def breed(self, scores, elite_count, tournament_size, sigma, mutation_rate=MUTATION_RATE):
        scores = torch.as_tensor(scores, dtype=torch.float32, device=self.device)
        order = scores.argsort(descending=True)
        child_count = self.count - elite_count

        candidates = torch.randint(self.count, (child_count, tournament_size), device=self.device)
        winners = scores[candidates].argmax(1, keepdim=True)
        parents = candidates.gather(1, winners).squeeze(1)
        parents[:elite_count] = order[:elite_count]

        child = Population.__new__(Population)
        child.count, child.sloj, child.device, child.dtype = self.count, self.sloj, self.device, self.dtype
        child.conv_weights = [
            self._breed_tensor(
                value, order, parents, elite_count, sigma / sqrt(in_channels * kernel * kernel), mutation_rate
            )
            for value, (in_channels, _, kernel, _) in zip(self.conv_weights, CONV_LAYERS)
        ]
        child.conv_biases = [
            self._breed_tensor(
                value, order, parents, elite_count, sigma / sqrt(in_channels * kernel * kernel), mutation_rate
            )
            for value, (in_channels, _, kernel, _) in zip(self.conv_biases, CONV_LAYERS)
        ]
        child.weights = [
            self._breed_tensor(value, order, parents, elite_count, sigma / sqrt(fan_in), mutation_rate)
            for value, fan_in in zip(self.weights, [DENSE_INPUT, *self.sloj[:-1]])
        ]
        child.biases = [
            self._breed_tensor(value, order, parents, elite_count, sigma / sqrt(fan_in), mutation_rate)
            for value, fan_in in zip(self.biases, [DENSE_INPUT, *self.sloj[:-1]])
        ]
        return child

    def _breed_tensor(self, values, order, parents, elite_count, step, mutation_rate):
        result = torch.empty_like(values)
        result[:elite_count] = values[order[:elite_count]]
        offspring = values[parents].clone()
        offspring.add_((torch.rand_like(offspring) < mutation_rate) * torch.randn_like(offspring) * step)
        result[elite_count:] = offspring
        return result


class ThreadedEnvs:
    def __init__(self, count, workers):
        self.envs = [make_env() for _ in range(count)]
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def reset(self, seed):
        futures = [self.pool.submit(env.reset, seed=seed) for env in self.envs]
        return np.stack([future.result()[0] for future in futures])

    def step(self, actions):
        futures = {
            index: self.pool.submit(env.step, actions[index])
            for index, env in enumerate(self.envs)
        }
        return {index: future.result() for index, future in futures.items()}

    def close(self):
        self.pool.shutdown()
        for env in self.envs:
            env.close()
