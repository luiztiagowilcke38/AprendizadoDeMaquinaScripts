from pygam import LinearGAM, s, f
import numpy as np

# Exemplo conceitual de GAM
# Use 's' para variaveis continuas (splines) 
# e 'f' para variaveis categoricas (factors)
X = np.random.randn(100, 2)
y = np.sin(X[:, 0]) + X[:, 1]**2 + np.random.randn(100) * 0.1

gam = LinearGAM(s(0) + s(1)).fit(X, y)

# A interpretabilidade e mantida examinando as funcoes parciais
# gam.summary()