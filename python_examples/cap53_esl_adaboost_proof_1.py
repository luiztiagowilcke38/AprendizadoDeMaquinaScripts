from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# AdaBoost classico usa Stumps (arvores de profundidade 1)
# como 'weak learners'
base = DecisionTreeClassifier(max_depth=1)
ada = AdaBoostClassifier(base_estimator=base, 
                         n_estimators=100, 
                         algorithm='SAMME')
# ada.fit(X, y)