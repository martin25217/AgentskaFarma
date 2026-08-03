from gym import Model
import random
from datetime import datetime
from pathlib import Path

kol = 100
velicina_populacije = 30
elitizam = int(velicina_populacije * 0.1)
broj_generacija = 100
sigma = 0.5
turnir_k = 3
seedovi = [0, 1, 2]
populacija= []


def turnir(evaluacija):
    return max(random.sample(evaluacija, turnir_k), key=lambda item: item[0])[1]

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
weights_dir = Path(__file__).parent / "weights"

for i in range (velicina_populacije):
    x = Model([kol,kol,kol,kol,kol,9], 33600)
    populacija.append(x)

for generacija in range(broj_generacija):
    evaluacija = []

    for model in populacija:
        score = sum(model.eval_model(seed=s) for s in seedovi) / len(seedovi)
        evaluacija.append([score, model])

    evaluacija.sort(key=lambda item: item[0], reverse= True)

    najbolji_score, najbolji = evaluacija[0]
    najbolji.save(weights_dir / f"{run_id}_gen_{generacija:03d}_score_{najbolji_score:g}.pkl")

    nova_populacija = []

    for idx in range(elitizam):
        nova_populacija.append(evaluacija[idx][1])
    
    for i in range (velicina_populacije - elitizam):
        p1=turnir(evaluacija)
        p2=turnir(evaluacija)

        dijete = Model([kol,kol,kol,kol,kol,9], 33600)
        dijete.cross(p1,p2,sigma)
        nova_populacija.append(dijete)

    print(generacija, najbolji_score)
    populacija = nova_populacija
    
    
