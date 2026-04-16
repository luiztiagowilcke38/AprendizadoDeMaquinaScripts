import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniCNN(nn.Module):
    def __init__(self):
        super(MiniCNN, self).__init__()
        # 1 canal de entrada (ex: tons de cinza), 16 filtros, kernel 3x3
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Reduz pela metade a resolucao espacial
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Assumindo entrada 28x28 (MNIST): 28/2/2 = 7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) # 10 classes finais

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7) # Achatar as features espaciais
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

print(MiniCNN())