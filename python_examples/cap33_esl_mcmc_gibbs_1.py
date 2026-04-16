import numpy as np

def gibbs_sampler_2d_gaussian(n_iter, rho):
    """
    Amostra de uma Normal Bivariada com correlacao rho 
    usando apenas as distribuicoes condicionais.
    """
    x = 0
    y = 0
    amostras = np.zeros((n_iter, 2))
    
    # Condiconais de uma normal bivariada:
    # x | y ~ N(rho*y, 1 - rho^2)
    # y | x ~ N(rho*x, 1 - rho^2)
    std = np.sqrt(1 - rho**2)
    
    for i in range(n_iter):
        x = np.random.normal(rho * y, std)
        y = np.random.normal(rho * x, std)
        amostras[i] = [x, y]
        
    return amostras

# O periodo de 'burn-in' deve ser descartado conforme 
# discutido no texto de ESL para garantir convergencia.