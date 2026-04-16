from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# Definicao dos modelos base (diversidade e chave para o sucesso do stacking)
estimators = [
    ('lr', RidgeCV()),
    ('svr', SVR()),
    ('rf', RandomForestRegressor(n_estimators=10))
]

# Meta-modelo: frequentemente uma regressao linear robusta
stacking = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())
# stacking.fit(X, y)