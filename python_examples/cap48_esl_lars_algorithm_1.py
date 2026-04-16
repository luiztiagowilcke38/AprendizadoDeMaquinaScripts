from sklearn.linear_model import lars_path
import matplotlib.pyplot as plt
import numpy as np

# Dados sinteticos
X = np.random.randn(50, 10)
y = X[:, 0] * 5 + X[:, 1] * 2 + np.random.normal(0, 0.1, 50)

# Calcula o caminho LARS
alphas, active, coefs = lars_path(X, y, method='lasso')

# coefs contem os coeficientes para cada passo do algoritmo.
# A visualizacao do 'path' mostra como as variaveis entram no modelo.