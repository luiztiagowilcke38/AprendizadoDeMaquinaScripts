import numpy as np

# Matriz de covariancia exemplo
S = np.array([[1, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 1]])
precision = np.linalg.inv(S)

# O ESL destaca que zeros aproximados indicam 
# independência condicional (aresta ausente no grafo).