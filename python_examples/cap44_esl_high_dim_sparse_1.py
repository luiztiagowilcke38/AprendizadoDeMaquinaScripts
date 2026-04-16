from sklearn.linear_model import ElasticNetCV
import numpy as np

# p=200, N=50 (High-Dim)
X = np.random.randn(50, 200)
y = X[:, 0] * 3 - X[:, 1] * 2 + np.random.normal(0, 0.1, 50)

# l1_ratio de 0.5 divide igualmente a penalidade L1 e L2
model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5)
model.fit(X, y)

# model.coef_ contera a maioria dos valores como zero.