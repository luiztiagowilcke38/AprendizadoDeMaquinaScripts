import numpy as np

def update_ising_node(grid, i, j, beta):
    """Reflete o espirito de Gibbs em uma rede de Markov"""
    vecinos = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
    s_sum = 0
    for r, c in vecinos:
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            s_sum += grid[r, c]
    # Probabilidade de ser +1 baseada nos vizinhos
    p_plus = 1 / (1 + np.exp(-2 * beta * s_sum))
    return 1 if np.random.rand() < p_plus else -1