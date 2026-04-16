import numpy as np
import matplotlib.pyplot as plt

def f(x): return x**2 + 10*np.sin(x)
def grad(x): return 2*x + 10*np.cos(x)

x = 8 # Ponto inicial
lr = 0.1
v = 0
beta = 0.9

history_gd = []
history_mom = []

# GD Puro
x_gd = x
for _ in range(50):
    history_gd.append(x_gd)
    x_gd -= lr * grad(x_gd)

# Momentum
x_mom = x
for _ in range(50):
    history_mom.append(x_mom)
    v = beta * v + (1 - beta) * grad(x_mom)
    x_mom -= lr * v

plt.plot(history_gd, label="GD Puro")
plt.plot(history_mom, label="Momentum")
plt.legend()
plt.title("Convergencia: GD vs Momentum")
plt.show()