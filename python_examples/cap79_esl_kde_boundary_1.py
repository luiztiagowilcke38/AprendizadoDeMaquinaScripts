import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# Dados estritamente positivos (Exponencial)
data = np.random.exponential(1, 1000)[:, None]
kde = KernelDensity(bandwidth=0.2).fit(data)

# Grid perto de zero
x_eval = np.linspace(-0.5, 2, 100)[:, None]
log_dens = kde.score_samples(x_eval)

# A densidade estimada sera > 0 para x < 0, 
# o que e impossivel para uma exponencial.