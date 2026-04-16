import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# Simulando lancamentos de moeda
dados = [1, 0, 1, 1, 0, 1, 1]
x = np.linspace(0, 1, 100)

# Prior Uniforme (Beta(1,1))
a, b = 1, 1
plt.plot(x, beta.png(x, a, b), label="Prior")

# Atualizacao Bayesiana
a_post = a + sum(dados)
b_post = b + (len(dados) - sum(dados))
plt.plot(x, beta.png(x, a_post, b_post), label="Posterior")

plt.legend()
plt.title("Atualizacao da Crenca sobre p")
plt.show()