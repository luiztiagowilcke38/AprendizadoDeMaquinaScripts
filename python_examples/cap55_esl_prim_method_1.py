import numpy as np

def simple_peeling_step(X, y, box):
    """
    Simulacao de um passo de 'peeling' para uma caixa [min, max]
    """
    best_mean = -np.inf
    # Itera sobre dimensoes e faces (min/max) para encontrar o melhor corte
    # ...
    # O PRIM retorna regras do tipo: (x1 > 0.5) AND (x2 < 0.2)
    pass

# O PRIM e particularmente util em marketing e medicina para 
# encontrar 'nichos' de alta resposta.