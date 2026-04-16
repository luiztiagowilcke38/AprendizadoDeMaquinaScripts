import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-2, 2, 100)
hinge = np.maximum(0, 1 - z)
log_loss = np.log2(1 + np.exp(-z))

# plt.plot(z, hinge, label='Hinge (SVM)')
# plt.plot(z, log_loss, label='Log-loss (Logistic)')
# No ESL, Figure 12.4 mostra como ambas aproximam a perda 0-1.