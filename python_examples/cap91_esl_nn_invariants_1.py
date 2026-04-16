from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# A distancia de Mahalanobis e uma forma de invariancia por escala
X = np.random.randn(100, 5)
y = np.random.randint(0, 2, 100)

knn = KNeighborsClassifier(metric='mahalanobis', 
                           metric_params={'VI': np.linalg.inv(np.cov(X.T))})
# knn.fit(X, y)