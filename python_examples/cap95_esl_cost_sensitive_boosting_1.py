from sklearn.ensemble import AdaBoostClassifier
import numpy as np

# Definindo pesos por amostra (Sample Weights) para classes desequilibradas
X = np.random.randn(100, 2)
y = np.array([0]*90 + [1]*10) # 90% classe 0

sample_weights = np.where(y == 1, 9.0, 1.0) # Penaliza 9x mais errar a classe 1
clf = AdaBoostClassifier().fit(X, y, sample_weight=sample_weights)

# O ESL destaca que isso e equivalente a mover a fronteira 
# de Bayes para compensar o custo esperado.