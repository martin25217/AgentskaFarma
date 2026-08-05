import tempfile
from pathlib import Path

import numpy as np
import torch

from gym import ACTION_COUNT, FRAME_STACK, PLAYER_ACTION_COUNT, PLAYERS, RAM_SIZE, Model, Population, make_env


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
        population.best(i)
        .feed_forward(observations[i])
        .reshape(len(PLAYERS), PLAYER_ACTION_COUNT)
        .argmax(1)
        for i in range(population.count)
    ],
)

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

env = make_env()
try:
    observation, _ = env.reset(seed=0)
    assert observation.shape == (FRAME_STACK, RAM_SIZE)
    assert all(env.env.action_space(player).n == PLAYER_ACTION_COUNT for player in PLAYERS)
finally:
    env.close()
print("OK")
