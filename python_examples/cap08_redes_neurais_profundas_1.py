import torch
import torch.nn as nn
import torch.optim as optim

# Definindo um MLP simples
class SimplesMLP(nn.Module):
    def __init__(self):
        super(SimplesMLP, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

modelo = SimplesMLP()
criterio = nn.BCELoss()
otimizador = optim.Adam(modelo.parameters(), lr=0.01)

# Treinamento Dummy
X = torch.randn(100, 2)
y = torch.randint(0, 2, (100, 1)).float()

for epoca in range(100):
    otimizador.zero_grad()
    saida = modelo(X)
    perda = criterio(saida, y)
    perda.backward() # A magica do Autograd acontece aqui
    otimizador.step()
    
print(f"Perda final: {perda.item():.4f}")