import numpy as np
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt

# Geracao de dados nao-lineares (senoidal + ruido)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(8))  # Adiciona outliers

# Ajuste do modelo Kernel Ridge (RBF Kernel)
# alpha = lambda no ESL
krr = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
krr.fit(X, y)

# Predicao para visualizacao
X_plot = np.linspace(0, 5, 100)[:, None]
y_plot = krr.predict(X_plot)

# No ESL, a escolha de gamma (largura do nucleo) e alpha 
# controla o trade-off vies-variancia discutido no Cap 21.