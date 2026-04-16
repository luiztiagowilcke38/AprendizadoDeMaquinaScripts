import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))

def irls_step(X, y, beta):
    p = sigmoid(X.dot(beta))
    W = np.diag(p * (1 - p))
    # Update rule
    hessian = X.T.dot(W).dot(X)
    grad = X.T.dot(y - p)
    return beta + np.linalg.inv(hessian).dot(grad)

# A convergencia e tipicamente alcancada em menos de 10 iteracoes.