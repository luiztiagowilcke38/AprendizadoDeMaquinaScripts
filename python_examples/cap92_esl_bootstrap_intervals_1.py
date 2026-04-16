import numpy as np

data = np.random.normal(10, 2, 50)
boots = []
for _ in range(1000):
    sample = np.random.choice(data, size=len(data), replace=True)
    boots.append(np.mean(sample))

# Intervalo percentil 95%
ic_low, ic_high = np.percentile(boots, [2.5, 97.5])
# print(f"IC: [{ic_low:.2f}, {ic_high:.2f}]")