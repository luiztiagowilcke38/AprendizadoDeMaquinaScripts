from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def seleciona_alpha_cv(X, y, alphas=None, cv=5):
    """
    Seleciona o alpha otimo de poda por validacao cruzada k-fold.
    Usa o caminho de poda Cost-Complexity.
    """
    arvore_base = DecisionTreeClassifier(random_state=42)
    path = arvore_base.cost_complexity_pruning_path(X, y)
    ccp_alphas = path.ccp_alphas[:-1]  # remove o alpha maximo (raiz)

    scores_cv = []
    for alpha in ccp_alphas:
        arvore = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
        score = cross_val_score(arvore, X, y, cv=cv, scoring='accuracy')
        scores_cv.append(score.mean())

    alpha_otimo = ccp_alphas[np.argmax(scores_cv)]
    return alpha_otimo, np.max(scores_cv)

# Exemplo de uso:
# from sklearn.datasets import load_iris
# X, y = load_iris(return_X_y=True)
# alpha_otimo, acuracia = seleciona_alpha_cv(X, y)
# print(f'Alpha otimo: {alpha_otimo:.4f}, Acuracia CV: {acuracia:.3f}')