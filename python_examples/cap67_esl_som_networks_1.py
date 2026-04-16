from minisom import MiniSom
import numpy as np

# Inicializando SOM 10x10 para dados 3D (ex: cores RGB)
som = MiniSom(10, 10, 3, sigma=1.0, learning_rate=0.5)
data = np.random.rand(100, 3) # Amostras de cores
som.train_random(data, 100)

# O ESL observa que o SOM e uma forma de Principal Curves 
# discretizada em uma grade topologica.