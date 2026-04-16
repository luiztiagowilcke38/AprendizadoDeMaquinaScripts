import numpy as np

# Regressao: Media reduz variancia consistentemente
erros_indiv = np.array([0.5, 0.4, 0.6])
# erro_media < media(erros_indiv) se houver independencia

# Classificacao: Cuidado com correccao de erros
# Se 3 modelos votam: {P[0, 1, 1]} y=0 -> Erro e 1.