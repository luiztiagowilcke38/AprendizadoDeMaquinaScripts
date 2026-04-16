import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Estrutura conceitual de um especialista linear
def linear_expert(x, beta):
    return x.dot(beta)

# O Gate decide a probabilidade de cada especialista
def gating_network(x, v):
    return softmax(x.dot(v))

# O modelo final e a soma ponderada:
# y_hat = sum(P(expert_i|x) * expert_i(x))