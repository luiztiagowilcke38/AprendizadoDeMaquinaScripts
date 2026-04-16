from sklearn.svm import SVR
import numpy as np

# SVR com nucleo RBF
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# epsilon define a largura do tubo insensivel
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X, y)

# No ESL, destaca-se que o SVR e robusto a outliers se C for moderado.