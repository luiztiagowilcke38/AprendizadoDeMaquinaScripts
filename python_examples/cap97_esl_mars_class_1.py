from pyearth import Earth
import numpy as np

# Para classificacao, pyearth pode ser usado com post_fit 
# ou transformando o problema em indicadores (dummy variables)
model = Earth(classifier=True)
# model.fit(X, y_binario)