from sklearn.decomposition import NMF
import numpy as np

# Exemplo com dados positivos (contagens)
V = np.abs(np.random.randn(10, 5))
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(V)
H = model.components_

# No ESL, Figura 14.33 mostra como NMF aprende 'bases' 
# parciais (ex: partes de rostos) enquanto PCA aprende bases globais.