import torch.nn as nn
import torch

# Exemplo de uma convolucao simples 3x3
conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
input_img = torch.randn(1, 1, 28, 28)
output = conv(input_img)

# No ESL, destaca-se que as CNNs sao a base para o Deep Learning moderno,
# automatizando a engenharia de caracteristicas (Feature Engineering).