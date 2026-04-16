from pyearth import Earth
import numpy as np

# MARS captura interacoes automaticamente
X = np.random.uniform(-1, 1, (100, 2))
y = X[:, 0] * X[:, 1] + 0.1 * np.random.randn(100)

model = Earth(max_degree=2) # permite interacoes de ordem 2
model.fit(X, y)

# O MARS reduz o numero de bases via um processo de 'pruning' 
# similar ao CART, usando GCV (Generalized Cross Validation).