import numpy as np

# Omega como matriz de segunda diferenca (suavizacao)
def second_diff_penalty(p):
    D = np.eye(p) - 2*np.eye(p, k=1) + np.eye(p, k=2)
    return D.T.dot(D)

# Integrar esta matriz no cálculo do discriminante regulariza o modelo.