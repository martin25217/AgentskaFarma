import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from gym import ACTION_COUNT, ROLLOUT_STEPS, Model, Population, ThreadedEnvs, resolve_device


def evaluate(population, envs, seeds, steps):
    total = np.zeros(population.count)
    for seed in seeds:
        observations = envs.reset(seed)
        scores = np.zeros(population.count)
        for _ in range(steps):
            actions = population.actions(observations)
            for index, (observation, reward, _, _, _) in envs.step(actions).items():
                observations[index] = observation
                scores[index] += reward
        total += scores
    return total / len(seeds)


def parse_args():
    parser = argparse.ArgumentParser(description="Evolve a cooperative two-paddle Pong policy on PyTorch.")
    parser.add_argument("--hours", type=float, default=1, help="maximum runtime (default: 1)")
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--steps", type=int, default=ROLLOUT_STEPS, help="policy decisions per seed")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--sigma-decay", type=float, default=0.995)
    parser.add_argument("--min-sigma", type=float, default=0.02)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--elite-fraction", type=float, default=0.1)
    parser.add_argument("--tournament-size", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    if not (
        args.hours > 0
        and args.population >= 2
        and args.width >= 1
        and args.steps >= 1
        and args.workers >= 1
        and args.seed
        and args.sigma > 0
        and 0 < args.sigma_decay <= 1
        and 0 < args.min_sigma <= args.sigma
        and 0 <= args.mutation_rate <= 1
        and 0 < args.elite_fraction < 1
        and args.tournament_size >= 2
    ):
        raise SystemExit("invalid training parameter; run with --help for the allowed ranges")

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(0)
    resumed = Model.load(args.resume, device) if args.resume else None
    if resumed and resumed.sloj[-1] != ACTION_COUNT:
        raise SystemExit("checkpoint is not a two-player Pong model")
    layers = resumed.sloj if resumed else [args.width] * 5 + [ACTION_COUNT]
    population = Population(args.population, layers, device)
    sigma = args.sigma
    best_score = float("-inf")
    generation = 0
    if resumed:
        population.inject(resumed)
        sigma = float(resumed.metadata.get("sigma", sigma))
        best_score = float(resumed.metadata.get("score", best_score))
        generation = int(resumed.metadata.get("generation", -1)) + 1

    elite_count = max(1, int(args.population * args.elite_fraction))
    deadline = time.monotonic() + args.hours * 3600
    checkpoint = Path(__file__).parent / "weights" / "pong-coop-best.pt"
    envs = ThreadedEnvs(args.population, min(args.workers, args.population))

    print(
        f"device={device} population={args.population} workers={min(args.workers, args.population)} steps={args.steps}",
        flush=True,
    )
    try:
        while time.monotonic() < deadline:
            scores = evaluate(population, envs, args.seed, args.steps)
            best_index = int(scores.argmax())
            generation_score = float(scores[best_index])
            if generation_score > best_score:
                best_score = generation_score
                population.best(best_index).save(
                    checkpoint, generation=generation, score=best_score, sigma=sigma, steps=args.steps
                )
                print(f"saved {checkpoint}", flush=True)

            print(
                f"generation={generation} best={generation_score:g} all_time={best_score:g} sigma={sigma:.4f}",
                flush=True,
            )
            if best_score == 0:
                print("perfect 0-point run; stopped", flush=True)
                break
            population = population.breed(
                scores, elite_count, args.tournament_size, sigma, args.mutation_rate
            )
            sigma = max(sigma * args.sigma_decay, args.min_sigma)
            generation += 1
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        envs.close()


if __name__ == "__main__":
    main()
