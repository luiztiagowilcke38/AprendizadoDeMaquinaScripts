from sklearn.linear_model import Ridge
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(1000, 20)
y = X @ np.random.rand(20) + np.random.normal(0, 0.5, 1000)

train_sizes, train_scores, test_scores = learning_curve(
    Ridge(), X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Treino")
plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Validacao")
plt.xlabel("Amostras de Treino")
plt.ylabel("Score R2")
plt.legend()
plt.show()