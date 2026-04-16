import torch
import torch.nn as nn
import torch.optim as optim

class AutoencoderLinear(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder (Comprime 784 -> 64)
        self.encoder = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64)
        )
        # Decoder (Reconstroi 64 -> 784)
        self.decoder = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 784), nn.Sigmoid()
        )

    def forward(self, x):
        latente = self.encoder(x)
        reconstrucao = self.decoder(latente)
        return reconstrucao

# Treinamento padrao
modelo = AutoencoderLinear()
criterio = nn.MSELoss() # Erro de reconstrucao
otimizador = optim.Adam(modelo.parameters(), lr=1e-3)

# Supondo `imagens` como Input [Batch, 784]
# saida = modelo(imagens)
# perda = criterio(saida, imagens) # O target e o proprio input!