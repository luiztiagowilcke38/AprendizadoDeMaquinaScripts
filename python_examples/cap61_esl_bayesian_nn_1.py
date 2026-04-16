import numpy as np

# Uma camada linear Bayesiana mantem media (mu) e variancia (sigma)
# para cada peso. No forward pass, amostramos w ~ N(mu, sigma)
def linear_bayesian_forward(x, mu, sigma):
    eps = np.random.normal(0, 1, mu.shape)
    w = mu + eps * sigma
    return x.dot(w)

# A perda inclui o termo KL-divergence entre q(theta) e o prior.
# O ESL destaca que isso regulariza a rede automaticamente.