import numpy as np
import scipy.optimize as opt

# Exemplo de aproximacao de Laplace para uma densidade beta-binomial
def log_posterior(theta, data):
    # theta deve estar em [0, 1]
    if theta < 0 or theta > 1: return -1e10
    n, k = data
    # Log-likelihood + Log-prior (Uniforme aqui)
    return k * np.log(theta) + (n - k) * np.log(1 - theta)

data = (10, 7) # 10 lancamentos, 7 caras
# Encontra a moda (MAP)
res = opt.minimize(lambda x: -log_posterior(x, data), x0=0.5)
map_est = res.x[0]

# Calcula a Hessiana numericamente
eps = 1e-4
hess = (log_posterior(map_est + eps, data) - 2*log_posterior(map_est, data) + 
        log_posterior(map_est - eps, data)) / eps**2
sigma_est = np.sqrt(-1/hess)

# A posterior e aproximada por N(map_est, sigma_est^2)