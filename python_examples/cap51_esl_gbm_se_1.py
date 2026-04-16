from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# Gradient Boosting com perda L2 (padrão)
X = np.random.rand(100, 1)
y = X**2 + np.random.normal(0, 0.1, (100, 1))

# 'ls' indica Least Squares (L2) no scikit-learn antigo, 
# nas versoes novas e automatico ou definido em 'loss'
gbm_l2 = GradientBoostingRegressor(loss='squared_error', 
                                  learning_rate=0.1, 
                                  n_estimators=100)
gbm_l2.fit(X, y.ravel())