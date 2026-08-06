from vizualizacijakordinati import run_genetic
from gym import cross
import itertools
import json
import time


POPULATION_SIZES = [20, 50, 100]
GENERATIONS_PER_TRIAL = [30]
SIGMAS = [0.2, 0.5, 0.8]
ELITISM_FRACTIONS = [0.05, 0.1, 0.2]  
MUTATION_PROBS = [0.8, 0.9, 0.98]

results = []

combos = list(itertools.product(
    POPULATION_SIZES, GENERATIONS_PER_TRIAL, SIGMAS, ELITISM_FRACTIONS, MUTATION_PROBS
))
print(f"Running {len(combos)} combinations...\n")

for velicina_populacije, broj_generacija, sigma, elitizam_frac, mutation_prob, in combos:
    elitizam = max(1, int(velicina_populacije * elitizam_frac))

    print(f"=== pop={velicina_populacije} gens={broj_generacija} "
          f"sigma={sigma} elitizam={elitizam} vjerojatnost_mutacije={mutation_prob} ===")

    t0 = time.time()
    best_score = run_genetic(
        sigma=sigma,
        velicina_populacije=velicina_populacije,
        elitizam=elitizam,
        broj_generacija=broj_generacija,
        mutation_prob=mutation_prob,
    )
    elapsed = time.time() - t0

    result = {
        "velicina_populacije": velicina_populacije,
        "broj_generacija": broj_generacija,
        "sigma": sigma,
        "elitizam": elitizam,
        "best_score": best_score,
        "mutation_prob": mutation_prob,
        "elapsed_seconds": round(elapsed, 1),
    }
    results.append(result)
    print(f"--> best_score={best_score}, took {elapsed:.1f}s\n")

    # Save progress after every trial in case the search is interrupted
    with open("grid_search_results.json", "w") as f:
        json.dump(results, f, indent=2)

results.sort(key=lambda r: r["best_score"], reverse=True)
print("\n=== Top 5 combinations ===")
for r in results[:5]:
    print(r)