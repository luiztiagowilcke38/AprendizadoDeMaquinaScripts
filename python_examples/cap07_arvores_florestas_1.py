from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

# Gerando dados complexos
X, y = make_moons(n_samples=500, noise=0.3, random_state=42)

# Treinamento
tree = DecisionTreeClassifier(max_depth=5).fit(X, y)
rf = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X, y)

# Visualizacao simplificada da fronteira
xx, yy = np.meshgrid(np.linspace(-2, 3, 100), np.linspace(-1.5, 2, 100))
Z_tree = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(100, 100)
Z_rf = rf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(100, 100)

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.contourf(xx, yy, Z_tree, alpha=0.4, cmap=plt.cm.RdBu)
plt.scatter(X[:,0], X[:,1], c=y, s=20, edgecolor='k')
plt.title("Arvore de Decisao (Overfitting local)")

plt.subplot(122)
plt.contourf(xx, yy, Z_rf, alpha=0.4, cmap=plt.cm.RdBu)
plt.scatter(X[:,0], X[:,1], c=y, s=20, edgecolor='k')
plt.title("Random Forest (Fronteira Suave)")
plt.show()