from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# Dados XOR-like
X = np.random.randn(200, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# SVM com Kernel RBF
clf = SVC(kernel='rbf', gamma=1.0)
clf.fit(X, y)

# Plotting
xx, yy = np.meshgrid(np.linspace(-3, 3, 500), np.linspace(-3, 3, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=10, cmap=plt.cm.RdBu, alpha=0.8)
plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k')
plt.title("SVM com Kernel RBF")
plt.show()