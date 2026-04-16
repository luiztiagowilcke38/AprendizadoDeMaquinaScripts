from sklearn.decomposition import FastICA
import numpy as np

# Geracao de sinais (uma onda quadrada e uma senoidal)
t = np.linspace(0, 8, 2000)
s1 = np.sin(2 * t)  
s2 = np.sign(np.sin(3 * t)) 
S = np.c_[s1, s2]

# Misturando os sinais
A = np.array([[1, 1], [0.5, 2]])
X = S @ A.T

# Recuperando fontes via ICA
ica = FastICA(n_components=2)
S_rec = ica.fit_transform(X)

# Nota: O ICA nao recupera a ordem ou o sinal (polaridade) 
# das fontes originais, conforme discutido no ESL.