import zlib
import numpy as np

# Dados simples (alta compressao)
data1 = "01" * 100
# Dados complexos (baixa compressao)
data2 = "".join(np.random.choice(["0", "1"], 200))

# print(len(zlib.compress(data1.encode())))
# print(len(zlib.compress(data2.encode())))
# Modelos que 'aprendem' o padrao de data1 teriam menor MDL.