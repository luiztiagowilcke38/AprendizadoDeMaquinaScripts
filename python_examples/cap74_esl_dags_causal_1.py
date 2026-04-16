# Simulacao conceitual de um DAG linear: X -> Y -> Z
X = np.random.normal(0, 1, 100)
Y = 0.5 * X + np.random.normal(0, 0.1, 100)
Z = 0.8 * Y + np.random.normal(0, 0.1, 100)

# Verificacao de independecia condicional:
# Corr(X, Z) e alta, mas Corr(X, Z | Y) aproxima-se de zero.