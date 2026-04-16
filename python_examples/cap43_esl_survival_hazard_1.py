from lifelines import CoxPHFitter
import pandas as pd

# Exemplo de Analise de Sobrevivencia
# 'E' indica evento (1) ou censura (0)
df = pd.DataFrame({
    'T': [10, 15, 20, 25, 30],
    'E': [1, 0, 1, 1, 0],
    'X': [0, 1, 0, 1, 0]
})

cph = CoxPHFitter()
cph.fit(df, duration_col='T', event_col='E')

# A interpretacao de exp(beta) e o 'Hazard Ratio'.
# Valores > 1 indicam maior risco (reducao do tempo de vida).