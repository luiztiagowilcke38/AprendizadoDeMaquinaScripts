from sklearn.manifold import TSNE, Isomap
import numpy as np

# Comparacao de Manifold Learning
X = np.random.rand(100, 10)

# Isomap (Baseado em grafo)
isomap = Isomap(n_neighbors=5, n_components=2).fit_transform(X)

# t-SNE (Probabilistico)
tsne = TSNE(n_components=2, perplexity=30.0).fit_transform(X)

# t-SNE e excelente para clusters mas nao preserva distancias globais,
# enquanto Isomap tenta preservar a estrutura global da variedade.