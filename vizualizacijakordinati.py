from gym import (
    Model, make_coordinate_env, COORDINATE_INPUTS,
    BIG_TILE_COORDINATES, CoordinateObservation, LoseLifeEndsRun,
)
import argparse
from pathlib import Path

import ale_py
import gymnasium as gym
import numpy as np
import random
import pickle as pc
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


gym.register_envs(ale_py)
ENTITY_NAMES = ("pacman", "orange", "cyan", "pink", "red", "fruit")
ENTITY_COLORS = ("yellow", "orange", "cyan", "pink", "red", "lime")
ENTITY_TEXT_COLORS = ("goldenrod", "darkorange", "darkcyan", "deeppink", "darkred", "green")
def visualize(seed=0, output=None):
    
    env = make_coordinate_env("rgb_array")
    observation, _ = env.reset(seed=seed)
    figure, (axis, coordinate_axis) = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={"width_ratios": (2, 1)})
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
        _   , _, terminated, _, info = env.step(0)
        assert terminated and env.env.lives == info["lives"]
    finally:
       env.close()
    print("OK")


def run_genetic(kol=100):
    populacija = []
    elitizam= 0.28155106641764793
    mutation_prob=0.8565541442800917
    sigma= 1.061292050731741
    velicina_populacije= 100

    for _ in range(velicina_populacije):
        x = Model([kol, int(kol/2), 30, 30, 20, 9], COORDINATE_INPUTS)
        populacija.append(x)

    for _ in range(broj_generacija):
        print("generacija: ", _)
        t0 = time.time()
        evaluacija = []
        for model in populacija:
            score = model.eval_model()
            evaluacija.append([score, model])
        t1 = time.time()
        print(f"eval phase: {t1-t0:.1f}s")

        t2 = time.time()
        print(f"full generation: {t2-t0:.1f}s", flush=True)

        evaluacija.sort(key=lambda item: item[0], reverse=True)
        if evaluacija[0][0] > maxi:
            maxi = evaluacija[0][0]
            pc.dump(evaluacija[0][1], open('peakmodel.pkl', 'wb'))
            print("updated", '\n')
        print("score: ", evaluacija[0][0])

        nova_populacija = []
        for idx in range(elitizam):
            nova_populacija.append(evaluacija[idx][1])

        for _ in range(velicina_populacije - elitizam):
            p1 = random.choice(evaluacija[:int(velicina_populacije/3)])[1]
            p2 = random.choice(evaluacija[:int(velicina_populacije/3)])[1]

            dijete = Model([kol, int(kol/2), int(kol/4), 30, 20, 9], COORDINATE_INPUTS)
            dijete.cross(p1, p2, sigma, mutation_prob)
            nova_populacija.append(dijete)

        populacija = nova_populacija
        sigma *= 0.995
        sigma = max(sigma, 0.02)

    temp_eval = []
    for model in populacija:
            score = model.eval_model()
            temp_eval.append([score, model])
    temp_eval.sort(key=lambda item: item[0], reverse=True)
    return temp_eval[0][0]
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ms. Pac-Man evolution or visualization.")
    parser.add_argument("--check", action="store_true", help="Run check on environment")
    parser.add_argument("--seed", type=int, default=0, help="Seed for visualization")
    parser.add_argument("--save", type=Path, metavar="FILE", help="Save visualization as GIF")
    
    parser.add_argument("--genetic", action="store_true", help="Run genetic algorithm instead of visualization")
    args = parser.parse_args()

    if args.genetic:
        run_genetic(
        sigma=1.061292050731741,
        velicina_populacije=50,
        elitizam=1,
        broj_generacija=2000,
        mutation_prob=0.8565541442800917, )
    else:
        if args.check:
            check()
        else:
            visualize(args.seed, args.save)


