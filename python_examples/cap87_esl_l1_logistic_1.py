from sklearn.linear_model import LogisticRegression
import numpy as np

# 'liblinear' suporta penalidade L1
clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
X = np.random.randn(100, 20)
y = np.random.randint(0, 2, 100)
clf.fit(X, y)

# clf.coef_ contera zeros para variaveis nao-informativas.