import numpy as np

def nuclear_norm_solve(Z, mask, lam, n_iter=10):
    X = np.zeros_like(Z)
    for _ in range(n_iter):
        # 1. Imputacao
        Z_filled = Z * mask + X * (1 - mask)
        # 2. SVD e limiarizacao de autovalores
        U, s, Vt = np.linalg.svd(Z_filled, full_matrices=False)
        s_soft = np.maximum(0, s - lam)
        X = U.dot(np.diag(s_soft)).dot(Vt)
    return X