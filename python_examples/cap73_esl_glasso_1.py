from sklearn.covariance import GraphicalLasso
import numpy as np

X = np.random.randn(50, 10)
# alpha e o lambda de penalizacao no ESL
model = GraphicalLasso(alpha=0.1)
model.fit(X)

matriz_precisao = model.precision_
# A esparsidade da matriz define a 'estratégia' do grafo.