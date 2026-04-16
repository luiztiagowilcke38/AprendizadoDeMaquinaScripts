from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Geracao de dados de alta dimensao com esparsidade
X, y = make_regression(n_samples=100, n_features=50, n_informative=5, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

modelos = {
    "Lasso": LassoCV(),
    "ElasticNet": ElasticNetCV(),
    "RandomForest": RandomForestRegressor(n_estimators=100),
    "GBM": GradientBoostingRegressor()
}

for nome, mod in modelos.items():
    mod.fit(X_train, y_train)
    mse = mean_squared_error(y_test, mod.predict(X_test))
    print(f"{nome:12}: MSE = {mse:.4f}")

# Como vimos no Cap. 28, a selecao final deve ser validada
# com rigor estatistico usando cross-validation aninhado.