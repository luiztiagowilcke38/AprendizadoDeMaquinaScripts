import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_features=10, n_informative=3)
clf = GradientBoostingClassifier().fit(X, y)

importancias = clf.feature_importances_
# plt.bar(range(10), importancias)
# Note que a soma das importancias e 1.0 (ou 100%).