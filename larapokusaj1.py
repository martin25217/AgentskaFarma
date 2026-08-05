import argparse
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np


gym.register_envs(ale_py)
ENTITY_NAMES = ("pacman", "orange", "cyan", "pink", "red", "fruit")
ENTITY_COLORS = ("yellow", "orange", "cyan", "pink", "red", "lime")
ENTITY_TEXT_COLORS = ("goldenrod", "darkorange", "darkcyan", "deeppink", "darkred", "green")
BIG_TILE_COORDINATES = np.array(((148, 146), (148, 14), (8, 147), (8, 15)))
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
# RAM layout: https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/mspacman.py


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
    """Return 6 characters/fruit, 4 big tiles, then 252 small-tile coordinate slots."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(-13, 256, (262, 2), np.float32)

    def observation(self, ram):
        coordinates = np.full((262, 2), -1, dtype=np.float32)
        coordinates[0] = ram[[10, 16]] + (-13, 1)

        if ram[47] > 0:
            coordinates[1] = ram[[6, 12]] + (-13, 1)
        for ghost in range(min(int(ram[19]), 3)):
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


def visualize(seed=0, output=None):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    env = make_coordinate_env("rgb_array")
    observation, _ = env.reset(seed=seed)
    figure, (axis, coordinate_axis) = plt.subplots(
        1, 2, figsize=(8, 6), gridspec_kw={"width_ratios": (2, 1)}
    )
    image = axis.imshow(env.render())
    visible = observation.copy()
    visible[(visible < 0).any(axis=1)] = np.nan
    character_points = axis.scatter(
        visible[:6, 0],
        visible[:6, 1],
        edgecolors=ENTITY_COLORS,
        s=90,
        facecolors="none",
        linewidths=2,
    )
    big_tile_points = axis.scatter(
        visible[6:10, 0],
        visible[6:10, 1],
        edgecolors="white",
        s=100,
        facecolors="none",
        linewidths=2,
    )
    small_tile_points = axis.scatter(
        visible[10:, 0],
        visible[10:, 1],
        edgecolors="white",
        s=20,
        facecolors="none",
        linewidths=0.7,
    )
    labels = [
        coordinate_axis.text(0, 0.85 - index * 0.12, name, color=color)
        for index, (name, color) in enumerate(zip(ENTITY_NAMES, ENTITY_TEXT_COLORS))
    ]
    big_tile_count = coordinate_axis.text(0, 0.13, "", color="black")
    small_tile_count = coordinate_axis.text(0, 0.01, "", color="black")
    axis.set_axis_off()
    coordinate_axis.set_title("Coordinates")
    coordinate_axis.set_axis_off()

    def update(_):
        nonlocal observation
        observation, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            observation, _ = env.reset()

        coordinates = observation
        visible = coordinates.copy()
        visible[(visible < 0).any(axis=1)] = np.nan
        image.set_data(env.render())
        character_points.set_offsets(visible[:6])
        big_tile_points.set_offsets(visible[6:10])
        small_tile_points.set_offsets(visible[10:])
        for name, label, (x, y) in zip(ENTITY_NAMES, labels, coordinates):
            label.set_text(
                f"{name:>6}: ({int(x):3}, {int(y):3})" if x >= 0 else f"{name:>6}: absent"
            )
        big_tile_count.set_text(f"big tiles: {np.count_nonzero(coordinates[6:10, 0] >= 0)}")
        small_tile_count.set_text(f"small tiles: {np.count_nonzero(coordinates[10:, 0] >= 0)}")
        axis.set_title(f"reward: {reward:g}")
        return (
            image,
            character_points,
            big_tile_points,
            small_tile_points,
            *labels,
            big_tile_count,
            small_tile_count,
        )

    animation = FuncAnimation(
        figure, update, frames=160, interval=50, blit=False, cache_frame_data=False
    )
    figure.canvas.mpl_connect("close_event", lambda _: env.close())
    if output:
        animation.save(output, writer="pillow", fps=20)
        env.close()
    else:
        plt.show()
    return animation


def check():
    env = make_coordinate_env("rgb_array")
    try:
        observation, _ = env.reset(seed=0)
        coordinates = observation
        assert observation.shape == (262, 2)
        assert np.array_equal(coordinates[0], [75, 99])
        assert np.array_equal(coordinates[6:10], BIG_TILE_COORDINATES)
        assert np.count_nonzero(coordinates[10:, 0] >= 0) == 150
        assert env.render().shape == (210, 160, 3)
        assert env.observation_space.contains(observation)
        env.env.lives += 1
        _, _, terminated, _, info = env.step(0)
        assert terminated and env.env.lives == info["lives"]
    finally:
        env.close()
    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RAM coordinates over Ms. Pac-Man.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save", type=Path, metavar="FILE", help="save the visualization as a GIF"
    )
    args = parser.parse_args()
    check() if args.check else visualize(args.seed, args.save)




        