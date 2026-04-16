import numpy as np

def ppr_projection_step(X, y_residual):
    """
    Encontra a melhor direcao w para projetar X 
    e ajusta uma spline g(w^T X)
    """
    # 1. Busca direcao omega
    # 2. Projetar: z = X.dot(omega)
    # 3. Suavizar: g = spline_fit(z, y_residual)
    pass

# O PPR e excelente para ignorar dimensoes irrelevantes, 
# pois projeta os dados em subespacos de baixa dimensao.