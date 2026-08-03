import pickle
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import torch


gym.register_envs(ale_py)
MUTATION_RATE = 0.1


class LoseLifeEndsRun(gym.Wrapper):
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.lives = info["lives"]
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        terminated = terminated or info["lives"] < self.lives
        return observation, reward, terminated, truncated, info


def make_env(render_mode=None):
    env = gym.make(
        "ALE/MsPacman-v5",
        obs_type="grayscale",
        render_mode=render_mode,
        repeat_action_probability=0.0,
        full_action_space=False,
    )
    return gym.wrappers.FlattenObservation(LoseLifeEndsRun(env))


def resolve_device(device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Install the CUDA build of PyTorch and check your NVIDIA driver.")
    return torch.device(device)


class Model:
    def __init__(self, sloj, ulaz, device="auto"):
        self.sloj = list(sloj)
        self.ulaz = ulaz
        self.device = resolve_device(device)
        self.weights = [
            torch.randn(fan_in, fan_out, device=self.device) / sqrt(fan_in)
            for fan_in, fan_out in zip([ulaz, *sloj[:-1]], sloj)
        ]
        self.biases = [torch.zeros(size, device=self.device) for size in sloj]

    @classmethod
    def from_tensors(cls, sloj, ulaz, weights, biases, device="auto", metadata=None):
        model = cls.__new__(cls)
        model.sloj = list(sloj)
        model.ulaz = ulaz
        model.device = resolve_device(device)
        model.weights = [torch.as_tensor(value, dtype=torch.float32, device=model.device) for value in weights]
        model.biases = [torch.as_tensor(value, dtype=torch.float32, device=model.device) for value in biases]
        model.metadata = metadata or {}
        return model

    def cross(self, x1, x2, sigma):
        self.weights = []
        self.biases = []
        for fan_in, a, b in zip([self.ulaz, *self.sloj[:-1]], x1.weights, x2.weights):
            self.weights.append(self._cross_tensor(a, b, sigma / sqrt(fan_in)))
        for a, b in zip(x1.biases, x2.biases):
            self.biases.append(self._cross_tensor(a, b, sigma))

    def _cross_tensor(self, a, b, step):
        a, b = a.to(self.device), b.to(self.device)
        child = torch.where(torch.rand_like(a) < 0.5, a, b)
        return child + (torch.rand_like(child) < MUTATION_RATE) * torch.randn_like(child) * step

    @torch.inference_mode()
    def feed_forward(self, observation):
        value = torch.as_tensor(observation, dtype=torch.float32, device=self.device).flatten()
        value = value / 127.5 - 1.0
        for weights, biases in zip(self.weights, self.biases):
            value = torch.tanh(value @ weights + biases)
        return value

    def eval_model(self, render_mode=None, seed=None):
        env = make_env(render_mode)
        try:
            observation, _ = env.reset(seed=seed)
            score = 0.0
            while True:
                action = int(self.feed_forward(observation).argmax().item())
                observation, reward, terminated, truncated, _ = env.step(action)
                score += reward
                if terminated or truncated:
                    return score
        finally:
            env.close()

    def save(self, path, **metadata):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "sloj": self.sloj,
            "ulaz": self.ulaz,
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
            with path.open("rb") as file:
                sloj, ulaz, weights, biases = pickle.load(file)
            return cls.from_tensors(sloj, ulaz, weights, biases, device)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return cls.from_tensors(
            checkpoint["sloj"],
            checkpoint["ulaz"],
            checkpoint["weights"],
            checkpoint["biases"],
            device,
            {key: value for key, value in checkpoint.items() if key not in {"sloj", "ulaz", "weights", "biases"}},
        )


class Population:
    def __init__(self, count, sloj, ulaz, device="auto"):
        self.count = count
        self.sloj = list(sloj)
        self.ulaz = ulaz
        self.device = resolve_device(device)
        self.weights = [
            torch.randn(count, fan_in, fan_out, device=self.device) / sqrt(fan_in)
            for fan_in, fan_out in zip([ulaz, *sloj[:-1]], sloj)
        ]
        self.biases = [torch.zeros(count, size, device=self.device) for size in sloj]

    def actions(self, observations):
        value = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        value = value / 127.5 - 1.0
        with torch.inference_mode():
            for weights, biases in zip(self.weights, self.biases):
                value = torch.tanh(torch.bmm(value.unsqueeze(1), weights).squeeze(1) + biases)
        return value.argmax(1).cpu().numpy()

    def best(self, index):
        return Model.from_tensors(
            self.sloj,
            self.ulaz,
            [value[index] for value in self.weights],
            [value[index] for value in self.biases],
            self.device,
        )

    def inject(self, model):
        for target, source in zip(self.weights, model.weights):
            target[0].copy_(source.to(self.device))
        for target, source in zip(self.biases, model.biases):
            target[0].copy_(source.to(self.device))

    def breed(self, scores, elite_count, tournament_size, sigma, mutation_rate=MUTATION_RATE):
        scores = torch.as_tensor(scores, dtype=torch.float32, device=self.device)
        order = scores.argsort(descending=True)
        child_count = self.count - elite_count

        candidates = torch.randint(self.count, (2, child_count, tournament_size), device=self.device)
        winners = scores[candidates].argmax(2, keepdim=True)
        parents = candidates.gather(2, winners).squeeze(2)

        child = Population.__new__(Population)
        child.count, child.sloj, child.ulaz, child.device = self.count, self.sloj, self.ulaz, self.device
        child.weights = [
            self._breed_tensor(value, order, parents, elite_count, sigma / sqrt(fan_in), mutation_rate)
            for value, fan_in in zip(self.weights, [self.ulaz, *self.sloj[:-1]])
        ]
        child.biases = [
            self._breed_tensor(value, order, parents, elite_count, sigma, mutation_rate)
            for value in self.biases
        ]
        return child

    def _breed_tensor(self, values, order, parents, elite_count, step, mutation_rate):
        result = torch.empty_like(values)
        result[:elite_count] = values[order[:elite_count]]
        a, b = values[parents[0]], values[parents[1]]
        offspring = torch.where(torch.rand_like(a) < 0.5, a, b)
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

    def step(self, actions, active):
        futures = {
            index: self.pool.submit(self.envs[index].step, int(actions[index]))
            for index in np.flatnonzero(active)
        }
        return {index: future.result() for index, future in futures.items()}

    def close(self):
        self.pool.shutdown()
        for env in self.envs:
            env.close()
