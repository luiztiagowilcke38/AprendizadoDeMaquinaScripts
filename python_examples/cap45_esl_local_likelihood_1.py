from sklearn.neighbors import KernelDensity
import numpy as np

# Estimativa de densidade de uma bimodal
x = np.concatenate([np.random.normal(0, 1, 100), 
                   np.random.normal(5, 1, 100)])[:, None]

kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(x)

# Calculo de log-probabilidade para um grid
x_plot = np.linspace(-5, 10, 1000)[:, None]
log_dens = kde.score_samples(x_plot)