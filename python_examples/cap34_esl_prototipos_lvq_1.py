import numpy as np

class LVQ:
    def __init__(self, n_prototypes_per_class=2, learning_rate=0.01):
        self.n_prototypes = n_prototypes_per_class
        self.lr = learning_rate

    def fit(self, X, y, epochs=100):
        classes = np.unique(y)
        # Inicializacao simples com k-means ou amostras aleatorias
        self.prototypes = []
        self.proto_labels = []
        for c in classes:
            c_indices = np.where(y == c)[0]
            chosen = np.random.choice(c_indices, self.n_prototypes)
            for idx in chosen:
                self.prototypes.append(X[idx])
                self.proto_labels.append(c)
        self.prototypes = np.array(self.prototypes)
        
        for _ in range(epochs):
            for i in range(len(X)):
                dists = np.linalg.norm(self.prototypes - X[i], axis=1)
                best_idx = np.argmin(dists)
                if self.proto_labels[best_idx] == y[i]:
                    self.prototypes[best_idx] += self.lr * (X[i] - self.prototypes[best_idx])
                else:
                    self.prototypes[best_idx] -= self.lr * (X[i] - self.prototypes[best_idx])