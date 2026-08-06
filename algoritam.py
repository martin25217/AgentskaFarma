from gym import Model, INPUT_DIM, napravi_species
import random as pyrandom
import pickle as pc
from numpy import *
import time
from collections import defaultdict
import copy

velicina_populacije = 100
elitizam = int(velicina_populacije * 0.1)
broj_generacija = 100
sigma = 0.5
populacija = []

def rebuild_graph(model):
    graph = defaultdict(list)

    for frm, to, weight, enabled, innovation, typ in model.weights:
        graph[to].append([
            frm,
            weight,
            enabled,
            innovation
        ])

    model.graf = graph

maxi = -10_000_000

base_model = Model(zeros(INPUT_DIM))

populacija = [
    copy.deepcopy(base_model)
    for _ in range(velicina_populacije)
]

for model in populacija:
    for gene in model.weights:
        gene[2] += random.normal(0, 0.1)
    rebuild_graph(model)

#for _ in range(velicina_populacije):
   # populacija.append(Model(zeros(INPUT_DIM)))


for generacija in range(broj_generacija):
    print("generacija:", generacija)
    t0 = time.time()

    evaluacija = []

    # 1. Evaluiraj cijelu populaciju
    for model in populacija:
        score = model.eval_model()
        model.score = score
        evaluacija.append([score, model])

    t1 = time.time()
    print(f"eval phase: {t1 - t0:.1f}s")

    species = napravi_species(populacija)

    print("broj speciesa:", len(species))

    for i, grupa in enumerate(species):
        print("species", i, "ima", len(grupa), "genoma")

    evaluacija.sort(key=lambda item: item[0], reverse=True)

    najbolji_score = evaluacija[0][0]
    najbolji_model = evaluacija[0][1]

    if najbolji_score > maxi:
        maxi = najbolji_score

        with open("peakmodel.pkl", "wb") as file:
            pc.dump(najbolji_model, file)

        print("updated\n")

    print(
        "top model id:",
        id(najbolji_model),
        "score:",
        najbolji_score
    )

    print("br neurona:", najbolji_model.actual_node_count())
    print("br veza:", len(najbolji_model.weights))

    nova_populacija = []

    for idx in range(elitizam):
        nova_populacija.append(evaluacija[idx][1])

    for _ in range(velicina_populacije - elitizam):
        p1 = pyrandom.choice(evaluacija[:max(1, velicina_populacije // 3)])[1]
        p2 = pyrandom.choice(evaluacija[:max(1, velicina_populacije // 3)])[1]

        dijete = Model.__new__(Model)
        dijete.score = 0
        dijete.cross(p1, p2, sigma)

        nova_populacija.append(dijete)

        print(
            id(dijete),
            len(dijete.weights),
            dijete.actual_node_count()
        )

    populacija = nova_populacija

    sigma *= 0.995
    sigma = max(sigma, 0.02)