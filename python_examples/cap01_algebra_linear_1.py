import numpy as np
import matplotlib.pyplot as plt

# Criando uma matriz de transformacao
A = np.array([[2, 1], [1, 2]])

# Criando um circulo unitario
theta = np.linspace(0, 2*np.pi, 100)
vecs = np.array([np.cos(theta), np.sin(theta)])

# Aplicando a transformacao
transformed_vecs = A @ vecs

# Visualizacao
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(vecs[0], vecs[1], 'b')
plt.title("Circulo Unitario")
plt.axis('equal')

plt.subplot(1,2,2)
plt.plot(transformed_vecs[0], transformed_vecs[1], 'r')
plt.title("Circulo Transformado (Elipse)")
plt.axis('equal')
plt.show()