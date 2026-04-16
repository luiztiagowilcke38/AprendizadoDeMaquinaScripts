from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
# n_components define o rank reduzido
lda = LinearDiscriminantAnalysis(n_components=2).fit(X, y)
X_r2 = lda.transform(X)

# plt.scatter(X_r2[:, 0], X_r2[:, 1], c=y)
# Mostra a separacao maxima entre as classes do Iris.