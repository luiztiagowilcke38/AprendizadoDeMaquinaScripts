from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# Introduzindo um outlier massivo
X = np.linspace(0, 10, 50)[:, None]
y = 2*X.ravel() + np.random.normal(0, 1, 50)
y[25] = 500 # Outlier

# Comparando L2 vs Huber
# Huber ignora o outlier extremo com mais facilidade
gbm_huber = GradientBoostingRegressor(loss='huber', alpha=0.9)
gbm_huber.fit(X, y)