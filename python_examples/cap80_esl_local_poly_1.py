import statsmodels.api as sm
import numpy as np

X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.1, 100)

# lowess utiliza pesos tri-cube e regressao linear local
lowess = sm.nonparametric.lowess(y, X, frac=0.2)