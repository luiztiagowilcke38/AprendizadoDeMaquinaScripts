from sklearn.manifold import MDS
import numpy as np

# MDS para visualizacao 2D
X = np.random.rand(50, 10) # 50 pontos em 10 dimensoes
mds = MDS(n_components=2, metric=True, random_state=42)
X_2d = mds.fit_transform(X)

# O MDS classico preserva distancias globais, 
# enquanto o Isomap (cap 42) foca em distancias geodesicas locais.