from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np

X = np.random.rand(100, 5)
scores = []
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k).fit(X)
    score = silhouette_score(X, kmeans.labels_)
    scores.append(score)

# O k que maximiza o Silhouette Score e frequentemente 
# o numero natural de grupos nos dados.