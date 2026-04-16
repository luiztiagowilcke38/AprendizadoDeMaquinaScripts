from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

X = np.random.rand(15, 2)
# Calcula a matriz de ligacao usando o metodo de Ward
Z = linkage(X, method='ward')

# O dendrograma permite visualizar a hierarquia e escolher
# o numero de clusters cortando a arvore em uma altura d.
# plt.figure()
# dendrogram(Z)