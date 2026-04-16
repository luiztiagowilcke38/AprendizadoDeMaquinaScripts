from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC
import numpy as np

X = np.random.randn(100, 5)
y = np.random.choice([0, 1, 2], 100) # 3 classes

# O sklearn gerencia OVA/OVO automaticamente para SVM
# mas podemos forcar a estrategia explicitamente:
ova_svm = OneVsRestClassifier(SVC(kernel='linear')).fit(X, y)
ovo_svm = OneVsOneClassifier(SVC(kernel='linear')).fit(X, y)

# Em redes neurais (cap 26), o multiclasse e nativo via Softmax.