from sklearn.linear_model import Lasso
import numpy as np
import matplotlib.pyplot as plt

X = np.random.randn(100, 10)
y = 3*X[:,0] - 2*X[:,1] + np.random.normal(0, 1, 100)

alphas = np.logspace(-4, 1, 100)
coefs = []

for a in alphas:
    lasso = Lasso(alpha=a)
    lasso.fit(X, y)
    coefs.append(lasso.coef_)

plt.plot(alphas, coefs)
plt.xscale('log')
plt.xlabel('Lambda (Alpha)')
plt.ylabel('Coeficientes')
plt.title('Caminho de Regularizacao LASSO')
plt.show()