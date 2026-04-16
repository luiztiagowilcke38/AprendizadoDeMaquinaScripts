import pandas as pd
from sklearn.impute import SimpleImputer

# Embora scikit-learn use imputadores globais,
# o CART original (R parsnip/rpart) implementa surrogates nativos.
df = pd.DataFrame({'X1': [1, 2, np.nan, 4], 'X2': [1.1, 2.1, 2.9, 4.1]})

# Imputacao baseada em correlacao (aproximacao do espirito do surrogate)
imp = SimpleImputer(strategy='mean')
X_corrigido = imp.fit_transform(df)

# No ESL, prova-se que surrogates exploram a estrutura local da arvore
# melhor do que imputacoes globais de media.