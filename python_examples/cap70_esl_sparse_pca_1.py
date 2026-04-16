from sklearn.decomposition import SparsePCA
import numpy as np

X = np.random.randn(100, 20)
# alpha controla a esparsidade das cargas
spca = SparsePCA(n_components=5, alpha=1.0)
X_reduced = spca.fit_transform(X)

# spca.components_ contera muitos zeros, indicando 
# selecao automatica de variaveis no espaco latente.