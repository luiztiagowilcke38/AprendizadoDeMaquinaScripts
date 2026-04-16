import numpy as np
from scipy.stats import ttest_ind

def benjamini_hochberg(p_values, q=0.05):
    """
    Implementacao do procedimento BH para controle de FDR.
    """
    m = len(p_values)
    p_sorted_idx = np.argsort(p_values)
    p_sorted = p_values[p_sorted_idx]
    
    # Condicao: p(k) <= (k/m) * q
    k_max = 0
    for k, p in enumerate(p_sorted, 1):
        if p <= (k / m) * q:
            k_max = k
            
    rejeitados = np.zeros(m, dtype=bool)
    if k_max > 0:
        rejeitados[p_sorted_idx[:k_max]] = True
    return rejeitados

# Exemplo conceitual:
# p_vals = np.random.uniform(0, 1, 1000) # 1000 testes nulos
# signif = benjamini_hochberg(p_vals, q=0.1) # Q-Value de 10%
# print(f"Hipoteses rejeitadas: {np.sum(signif)}")