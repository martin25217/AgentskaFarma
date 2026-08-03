import tempfile
from pathlib import Path

import numpy as np
import torch

from gym import Model, Population


layers = [4, 3]
population = Population(6, layers, 5, "cpu")
actions = population.actions(np.zeros((6, 5), dtype=np.uint8))
assert actions.shape == (6,)

child = population.breed(np.arange(6), elite_count=1, tournament_size=3, sigma=0.5)
assert all(torch.equal(a[0], b[5]) for a, b in zip(child.weights, population.weights))
assert any(not torch.equal(a[1:], b[1:]) for a, b in zip(child.weights, population.weights))

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "best.pt"
    population.best(5).save(path, score=42)
    loaded = Model.load(path, "cpu")

assert loaded.sloj == layers and loaded.ulaz == 5
assert all(torch.equal(a, b[5]) for a, b in zip(loaded.weights, population.weights))
print("OK")
