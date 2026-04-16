import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

# Exemplo de reducao de variancia via Bagging
X = np.linspace(0, 5, 100)[:, None]
y = np.sin(X).ravel() + np.random.normal(0, 0.5, 100)

base = DecisionTreeRegressor()
bagging = BaggingRegressor(base_estimator=base, n_estimators=50).fit(X, y)

# A variancia da predição individual e muito maior que a do ensemble.
# No ESL, prova-se que o Bagging nao pode aumentar o vies se o 
# estimador base for linear, mas pode em estimadores nao-lineares.