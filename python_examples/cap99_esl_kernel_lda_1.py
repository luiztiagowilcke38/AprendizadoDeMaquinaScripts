from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.kernel_approximation import RBFSampler

# Aproximacao do Kernel LDA via amostragem de caracteristicas
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X)
klda = LinearDiscriminantAnalysis().fit(X_features, y)

# No ESL, Figura 12.13 compara fronteiras de SVM vs Kernel-LDA.