import sys
from numpy import *

rng = random.default_rng()

sys.path.pop(0)

import ale_py
import gymnasium as gym


gym.register_envs(ale_py)

BIG_TILE_COORDINATES = array(((148, 146), (148, 14), (8, 147), (8, 15)))
BIG_TILE_BITS = (4, 8, 16, 32)
SMALL_TILE_RAM_BITS = (
    (0, 64), (1, 128), (1, 32), (1, 8), (1, 2), (2, 1), (2, 4), (2, 16), (2, 64),
    (1, 64), (1, 16), (1, 4), (1, 1), (2, 2), (2, 8), (2, 32), (2, 128), (0, 16),
)
SMALL_TILE_GRIDS = tuple(
    tuple(grid.splitlines())
    for grid in (
        """111101111111101111
100101000000101001
111111111111111111
010100100001001010
110111111111111011
010000100001000010
011111100001111110
010000100001000010
110111111111111011
010101010010101010
111101011110101111
100101110011101001
100100010010001001
111111111111111111""",
        """111111111111111111
100001000000100001
110111011110111011
010101010010101010
010101111111101010
111100100001001111
100111100001111001
100100100001001001
100101111111101001
111111000000111111
001001011110100100
111101110011101111
100101000000101001
111101111111101111""",
    )
)

class LoseLifeEndsRun(gym.Wrapper):
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.lives = info["lives"]
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        lives = info["lives"]
        terminated = terminated or lives < self.lives
        self.lives = lives
        return observation, reward, terminated, truncated, info

class CoordinateObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(-13, 256, (262, 2), float32)

    def observation(self, ram):
        coordinates = full((262, 2), -1, dtype=float32)
        coordinates[0] = ram[[10, 16]] + (-13, 1)

        if ram[47] > 0:
            coordinates[1] = ram[[6, 12]] + (-13, 1)
        ghosts = int(ram[19])
        for ghost in range(ghosts if ghosts < 3 else 3):
            coordinates[ghost + 2] = ram[[ghost + 7, ghost + 13]] + (-13, 1)

        if ram[11] > 0 and ram[17] > 0:
            coordinates[5] = ram[[11, 17]] + (-13, 1)

        for index, (position, bit) in enumerate(zip(BIG_TILE_COORDINATES, BIG_TILE_BITS), 6):
            if ram[117] & bit:
                coordinates[index] = position

        grid = SMALL_TILE_GRIDS[int(ram[0] != 0)]
        for row, grid_row in enumerate(grid):
            for column, (ram_offset, bit) in enumerate(SMALL_TILE_RAM_BITS):
                if grid_row[column] == "1" and ram[59 + row * 3 + ram_offset] & bit:
                    coordinates[10 + row * 18 + column] = (
                        8 + column * 8 + (4 if column >= 9 else 0),
                        7 + row * 12,
                    )
        return coordinates

def make_coordinate_env(render_mode=None):
    env = gym.make(
        "ALE/MsPacman-v5",
        obs_type="ram",
        render_mode=render_mode,
        repeat_action_probability=0.0,
        full_action_space=False,
    )
    return CoordinateObservation(LoseLifeEndsRun(env))

COORDINATE_INPUTS = 262 * 2
env = make_coordinate_env()


class Model:
    def cross(self, x1, x2, sigma, mutation_prob):
        self.sigma = sigma

        weightx = []
        biasex = []

        for w1, w2 in zip(x1.weights, x2.weights):
            nw = (w1 + w2) / 2
            mask = rng.random(nw.shape) > 0.85
            nw = nw + mask * rng.normal(0, sigma, nw.shape)
            weightx.append(nw)

        for b1, b2 in zip(x1.biases, x2.biases):
            nb = (b1 + b2) / 2
            mask = rng.random(nb.shape) > mutation_prob
            nb += mask * rng.normal(0, sigma, nb.shape)
            biasex.append(nb)

        self.weights = weightx
        self.biases = biasex


    def feed_forward(self, ulaz):
        ulaz = ulaz.ravel() / 127.5 - 1.0

        for weights, biases in zip(self.weights, self.biases):
            ulaz = tanh(matmul(ulaz, weights) + biases)

        return ulaz
    
    def __init__(self, sloj, ulaz):
        self.sloj = sloj
        self.ulaz = ulaz
        weights = []
        biases = []
        prev = ulaz
        for size in sloj:
            weights.append(rng.normal(0, 1/sqrt(prev), size=(prev, size)))
            biases.append(zeros(size))
            prev = size
        self.weights = weights
        self.biases = biases

 
    def eval_model(self):
        obs, info = env.reset()
        poc = info["lives"]
        result = 0
        prev_obs = obs.copy()
        while True:
            action = int(argmax(self.feed_forward(obs)))
            obs, reward, terminated, truncated, info = env.step(action)

            prev_obs = obs.copy()
            result = result + reward
            if terminated or truncated or info["lives"] < poc:
                break
        return result
    
