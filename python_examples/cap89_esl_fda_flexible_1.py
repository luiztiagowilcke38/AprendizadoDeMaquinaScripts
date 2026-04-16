# No scikit-learn, o FDA pode ser simulado usando 
# bases de expansao (PolynomialFeatures) antes do LDA 
# ou usando regressao multivariada seguida de classificacao.
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

fda_sim = Pipeline([
    ('expand', PolynomialFeatures(degree=2)),
    ('lda', LinearDiscriminantAnalysis())
])
# fda_sim.fit(X, y)