import tempfile
from pathlib import Path

import numpy as np
import torch

from algoritam import evaluate
from gym import (
    ACTION_COUNT,
    FRAME_STACK,
    PLAYER_ACTION_COUNT,
    PLAYERS,
    PONG_ACTIONS,
    RAM_SIZE,
    Model,
    Population,
    make_env,
    tracking_reward,
)


class EvaluationStub:
    count = 2

    def actions(self, observations):
        return np.zeros((len(observations), len(PLAYERS)), dtype=int)


class EnvStub:
    def reset(self, seed):
        return np.zeros((2, 1))

    def step(self, actions):
        return {index: (np.zeros(1), float(index + 1), False, False, {}) for index in range(2)}


assert np.array_equal(evaluate(EvaluationStub(), EnvStub(), [0], 3), [3, 6])

layers = [4, ACTION_COUNT]
population = Population(6, layers, "cpu")
observations = np.random.default_rng(0).integers(
    256, size=(6, FRAME_STACK, RAM_SIZE), dtype=np.uint8
)
actions = population.actions(observations)
assert actions.shape == (6, len(PLAYERS))
assert np.array_equal(
    actions,
    [
        PONG_ACTIONS[
            population.best(i)
            .feed_forward(observations[i])
            .reshape(len(PLAYERS), PLAYER_ACTION_COUNT)
            .argmax(1)
            .cpu()
            .numpy()
        ]
        for i in range(population.count)
    ],
)
opponents = np.roll(np.arange(population.count), 1)
opponent_actions = population.actions(observations, opponents)
assert np.array_equal(
    opponent_actions,
    [
        PONG_ACTIONS[
            population.best(opponents[i])
            .feed_forward(observations[i])
            .reshape(len(PLAYERS), PLAYER_ACTION_COUNT)
            .argmax(1)
            .cpu()
            .numpy()
        ]
        for i in range(population.count)
    ],
)

assert tracking_reward([80, 100], ball_x=150, ball_y=80, ball_dx=1) > tracking_reward(
    [100, 100], ball_x=150, ball_y=80, ball_dx=1
)
assert tracking_reward([100, 80], ball_x=100, ball_y=80, ball_dx=-1) > tracking_reward(
    [100, 100], ball_x=100, ball_y=80, ball_dx=-1
)
assert tracking_reward([80, 80], ball_x=100, ball_y=80, ball_dx=1) == 0

child = population.breed(np.arange(6), elite_count=1, tournament_size=3, sigma=0.5)
assert all(
    torch.equal(a[0], b[5])
    for a, b in zip(child.weights, population.weights)
)
assert any(not torch.equal(a[1:], b[1:]) for a, b in zip(child.weights, population.weights))

clones = population.breed(
    np.arange(6), elite_count=1, tournament_size=3, sigma=0.5, mutation_rate=0
)
child_tensors = [*clones.weights, *clones.biases]
parent_tensors = [*population.weights, *population.biases]
assert all(torch.equal(child[1], parent[5]) for child, parent in zip(child_tensors, parent_tensors))
assert all(
    any(
        all(
            torch.equal(child[index], parent[parent_index])
            for child, parent in zip(child_tensors, parent_tensors)
        )
        for parent_index in range(population.count)
    )
    for index in range(1, population.count)
)

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "best.pt"
    population.best(5).save(path, score=42)
    loaded = Model.load(path, "cpu")

assert loaded.sloj == layers
assert all(torch.equal(a, b[5]) for a, b in zip(loaded.weights, population.weights))
resumed = Population(6, layers, "cpu")
resumed.inject(loaded)
assert all(torch.equal(row, source[5].expand_as(row)) for row, source in zip(resumed.weights, population.weights))

env = make_env()
try:
    observation, _ = env.reset(seed=0)
    assert observation.shape == (FRAME_STACK, RAM_SIZE)
    assert all(max(PONG_ACTIONS) < env.env.action_space(player).n for player in PLAYERS)
    _, reward, _, _, _ = env.step([PONG_ACTIONS[0]] * len(PLAYERS))
    assert isinstance(reward, float)
finally:
    env.close()
print("OK")
