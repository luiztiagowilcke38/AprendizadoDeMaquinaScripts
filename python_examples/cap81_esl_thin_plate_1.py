from scipy.interpolate import Rbf
import numpy as np

# Dados 2D -> 1D
x = np.random.rand(10, 2)
z = np.sin(x[:, 0]) * np.cos(x[:, 1])

# rbf com funcao 'thin_plate'
itp = Rbf(x[:, 0], x[:, 1], z, function='thin_plate')

# Predicao em grid
# x_new = np.meshgrid(...)
# z_new = itp(x_grid, y_grid)