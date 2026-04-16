from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Aprox. da matriz de proximidade via apply()
rf = RandomForestClassifier(n_estimators=100).fit(X, y)
folhas = rf.apply(X) # Retorna indices das folhas para cada amostra

prox_matrix = np.zeros((len(X), len(X)))
for m in range(100):
    for i in range(len(X)):
        prox_matrix[i, :] += (folhas[:, m] == folhas[i, m])

# Matriz normalizada
prox_matrix /= 100
# Valores proximos a 1 indicam forte similaridade estrutural.