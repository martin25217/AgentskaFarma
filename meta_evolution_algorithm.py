
from vizualizacijakordinati import run_genetic
import random
import json
import time
import copy



META_POPULATION_SIZE = 12
META_GENERATIONS = 15
META_ELITISM = 2        
GENERATIONS_PER_TRIAL = 30  

VELICINA_POPULACIJE = 100  


BOUNDS = {
    "elitizam_frac":       (0.02, 0.3),   
    "mutation_prob":       (0.3, 0.98),   
    "sigma":               (0.9, 1.3)
}

MUTATION_STRENGTH = {
    "elitizam_frac":       0.03,
    "mutation_prob":       0.08,
    "sigma":               0.1,
}


def random_individual():
    return {
        "elitizam_frac": random.uniform(*BOUNDS["elitizam_frac"]),
        "mutation_prob": random.uniform(*BOUNDS["mutation_prob"]),
        "sigma": random.uniform(*BOUNDS["sigma"]),
    }


def clip(value, key):
    lo, hi = BOUNDS[key]
    return max(lo, min(hi, value))


def mutate(individual):
    child = copy.deepcopy(individual)
    for key in child:
        if random.random() < 0.5:  
            child[key] += random.gauss(0, MUTATION_STRENGTH[key])
            child[key] = clip(child[key], key)
    return child


def crossover(parent1, parent2):
    child = {}
    for key in parent1:

        t = random.random()
        child[key] = parent1[key] * t + parent2[key] * (1 - t)
        child[key] = clip(child[key], key)
    return child


def evaluate(individual):
    elitizam = max(1, int(VELICINA_POPULACIJE * individual["elitizam_frac"]))

    print(f"  eval: pop={VELICINA_POPULACIJE} elitizam={elitizam} "
          f"sigma={individual['sigma']:.3f} mutation_prob={individual['mutation_prob']:.3f}")

    t0 = time.time()
    score = run_genetic(
        sigma=individual["sigma"],
        velicina_populacije=VELICINA_POPULACIJE,
        elitizam=elitizam,
        broj_generacija=GENERATIONS_PER_TRIAL,
        mutation_prob=individual["mutation_prob"],
    )
    elapsed = time.time() - t0
    print(f"  --> score={score}, took {elapsed:.1f}s")
    return score, elapsed


def run_meta_evolution():
    population = [random_individual() for _ in range(META_POPULATION_SIZE)]
    all_results = []

    for meta_gen in range(META_GENERATIONS):
        print(f"\n=== Meta-generation {meta_gen} ===")

        evaluated = []
        for individual in population:
            score, elapsed = evaluate(individual)
            record = {**individual, "velicina_populacije": VELICINA_POPULACIJE,
                       "score": score, "elapsed_seconds": round(elapsed, 1),
                       "meta_generation": meta_gen}
            evaluated.append((score, individual))
            all_results.append(record)

            with open("meta_evolution_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

        evaluated.sort(key=lambda pair: pair[0], reverse=True)
        best_score = evaluated[0][0]
        print(f"Best score this meta-generation: {best_score}")
        print(f"Best hyperparameters: {evaluated[0][1]}")


        next_population = []

        for i in range(META_ELITISM):
            next_population.append(copy.deepcopy(evaluated[i][1]))


        parent_pool = [ind for _, ind in evaluated[:max(2, META_POPULATION_SIZE // 3)]]
        while len(next_population) < META_POPULATION_SIZE:
            p1, p2 = random.sample(parent_pool, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            next_population.append(child)

        population = next_population

    all_results.sort(key=lambda r: r["score"], reverse=True)
    print("\n=== Top 5 hyperparameter sets overall ===")
    for r in all_results[:5]:
        print(r)

    return all_results


if __name__ == "__main__":
    run_meta_evolution()