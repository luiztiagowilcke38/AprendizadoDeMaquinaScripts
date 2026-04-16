from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# Comparacao de interacoes: max_depth=1 (aditivo puro) vs max_depth=3
gbm_stump = GradientBoostingRegressor(max_depth=1).fit(X, y)
gbm_tree = GradientBoostingRegressor(max_depth=3).fit(X, y)

# No ESL, recomenda-se 4 <= J <= 8 para a maioria dos problemas 
# de mineracao de dados tabular.