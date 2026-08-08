import numpy as np
import matplotlib.pyplot as plt

# Podaci
x = np.arange(1, 10.5, 0.5)
y = [2, 5, 4, 3.5, 7, 6, 2, 6, 8, 3, 5, 8, 5, 0, 1, 4, 2, 6, 9] 

# Linijski graf
plt.plot(x, y, color="green", label="Linija", linewidth=2, linestyle='-.', "alpha je 0.5")

# Točke
plt.scatter(x, y, color="red", s=160, label="Točke", marker='x', edgecolors='black',)

# Naslov i nazivi osi
plt.title("Graf bodova")
plt.xlabel("vrijeme")
plt.ylabel("bodovi")

# Mreža i legenda
plt.grid(True)
plt.legend()

# Prikaz grafa
plt.show()
