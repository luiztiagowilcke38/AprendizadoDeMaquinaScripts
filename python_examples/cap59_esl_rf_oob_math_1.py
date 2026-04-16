from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# rf = RandomForestRegressor(oob_score=True, n_estimators=200)
# rf.fit(X, y)
# print(f"Erro OOB: {1 - rf.oob_score_}")

# No ESL, prova-se que o erro OOB e quase identico 
# ao erro obtido por cross-validation N-fold.