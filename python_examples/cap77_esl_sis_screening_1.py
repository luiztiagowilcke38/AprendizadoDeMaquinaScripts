import numpy as np

def simple_sis(X, y, d):
    """Filtra as d variaveis com maior correlacao absoluta"""
    corrs = np.abs(np.array([np.corrcoef(X[:, j], y)[0,1] for j in range(X.shape[1])]))
    top_indices = np.argsort(corrs)[-d:]
    return X[:, top_indices], top_indices

# Este passo precede o uso de Lasso ou Elastic Net (Cap. 44).