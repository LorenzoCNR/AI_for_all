import numpy as np
import matplotlib.pyplot as plt

# Dati di esempio: traiettoria (x, y)
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Calcolo di angolo (theta) e distanza (r)
r = np.sqrt(x**2 + y**2)
theta = np.arctan2(y, x)

# Ricostruzione di (x, y) da r e theta
x_rec = r * np.cos(theta)
y_rec = r * np.sin(theta)

# Plot della traiettoria originale e ricostruita
plt.figure(figsize=(8, 8))
plt.plot(x, y, label='Traiettoria Originale')
plt.plot(x_rec, y_rec, '--', label='Traiettoria Ricostruita')
plt.legend()
plt.title("Traiettoria Originale e Ricostruita")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

# Plot di raggio e angolo
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(r, label='Distanza r (dal centro)')
ax[0].set_title("Distanza r")
ax[1].plot(theta, label='Angolo θ')
ax[1].set_title("Angolo θ")
plt.tight_layout()
plt.show()