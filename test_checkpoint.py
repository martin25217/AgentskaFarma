import tempfile
from pathlib import Path

from numpy import array_equal

from gym import Model

kol = 3
sloj = [kol, kol, kol, kol, kol, 9]

x1 = Model(sloj, 33600)
x2 = Model(sloj, 33600)

dijete = Model(sloj, 33600)
dijete.cross(x1, x2, 0.5)
assert [w.shape for w in dijete.weights] == [w.shape for w in x1.weights]
assert [b.shape for b in dijete.biases] == [b.shape for b in x1.biases]
assert not any(w.dtype == object for w in dijete.weights), "mutacija vraca array umjesto broja"

brat = Model(sloj, 33600)
brat.cross(x1, x2, 0.5)
assert not array_equal(brat.weights[0], dijete.weights[0]), "tezine se ne mutiraju"

djeca = []
for _ in range(5):
    d = Model(sloj, 33600)
    d.cross(x1, x2, 0.5)
    djeca.append(d)
assert any(b.any() for d in djeca for b in d.biases), "biasi se ne mutiraju"

assert dijete.eval_model(seed=0) == dijete.eval_model(seed=0)

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "sub" / "test.pkl"
    dijete.save(path)
    ucitan = Model.load(path)

assert ucitan.sloj == dijete.sloj and ucitan.ulaz == dijete.ulaz
assert all(array_equal(a, b) for a, b in zip(ucitan.weights, dijete.weights))
assert all(array_equal(a, b) for a, b in zip(ucitan.biases, dijete.biases))

prvi = dijete.eval_model()
drugi = ucitan.eval_model()
print("score:", prvi, drugi)

print("OK")
