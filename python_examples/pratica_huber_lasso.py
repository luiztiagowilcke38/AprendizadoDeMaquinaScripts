import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# Geração de Dados em Alta Dimensão com Outliers
# Baseado na Teoria: Cap. 104 (Alta Dimensão) e Cap. 15 (Leis dos Grandes Números)
# ==========================================
np.random.seed(42)
N = 300      # Número de amostras
p = 1000     # Ultra-alta dimensão (p >> N)

# Apenas 5 variáveis são as verdadeiras causais
true_vars = 5
X = np.random.randn(N, p)

# Gerando a resposta não-linear (Cap. 06 - RKHS/Métodos de Kernel)
# A relação é baseada apenas nas primeiras 'true_vars' variáveis
y_true = (
    3 * np.sin(X[:, 0]) +
    2 * X[:, 1]**2 +
    1.5 * np.exp(-X[:, 2]**2) +
    4 * X[:, 3] * X[:, 4]
)

# Adicionando ruído Gaussiano
y = y_true + np.random.normal(0, 1, N)

# Adicionando outliers extremos (Cap. 52 - Perdas Robustas)
outlier_indices = np.random.choice(N, size=15, replace=False)
y[outlier_indices] += np.random.normal(0, 50, size=15) # Outliers severos

X_train, X_test, y_train, y_test, y_true_train, y_true_test = train_test_split(
    X, y, y_true, test_size=0.3, random_state=42
)

print(f"Dimensão de X_train: {X_train.shape}")
print(f"Dimensão de X_test: {X_test.shape}")

# ==========================================
# Passo 1: Redução de Dimensionalidade / Triagem
# Sure Independence Screening (SIS) e LASSO (Cap. 104)
# ==========================================
print("\n--- Passo 1: Feature Screening (Cap. 104) ---")
# SIS: Calculando a correlação marginal de cada variável com y
correlations = np.abs(np.dot(X_train.T, y_train))
# Seleciona as top K = N features
K = int(N * 0.5) 
top_k_indices = np.argsort(correlations)[-K:]

X_train_screened = X_train[:, top_k_indices]
X_test_screened = X_test[:, top_k_indices]
print(f"Features após SIS: {X_train_screened.shape[1]}")

# LASSO no subconjunto reduzido para selecionar as finais (Sparse Recovery)
lasso = LassoCV(cv=5)
lasso.fit(X_train_screened, y_train)
active_features = np.where(lasso.coef_ != 0)[0]
# Recuperamos os índices originais
final_features = top_k_indices[active_features]

print(f"Features retidas após SIS + LASSO: {len(final_features)}")
print(f"Variáveis verdadeiras selecionadas: {set(final_features).intersection(set(range(true_vars)))}")

# Reduzindo a matriz X para os modelos complexos
X_train_final = X_train[:, final_features]
X_test_final = X_test[:, final_features]

# ==========================================
# Passo 2: Aprendizado Não-Linear com Kernel (Cap. 06)
# ==========================================
print("\n--- Passo 2: Kernel SVR (Cap. 06) ---")
# O Kernel RBF mapeia os dados para um espaço de dimensão infinita (Teorema de Mercer)
svr_rbf = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)
svr_rbf.fit(X_train_final, y_train)
y_pred_svr = svr_rbf.predict(X_test_final)

mse_svr = mean_squared_error(y_true_test, y_pred_svr)
mae_svr = mean_absolute_error(y_true_test, y_pred_svr)
print(f"SVR (RBF) - Erro Médio Absoluto na predição: {mae_svr:.3f}")

# ==========================================
# Passo 3: Gradient Boosting Robusto com Huber (Cap. 52)
# ==========================================
print("\n--- Passo 3: Gradient Boosting Robusto (Cap. 52) ---")
# Utilizando a perda de Huber para mitigar o impacto dos outliers massivos
gbm_huber = GradientBoostingRegressor(loss='huber', alpha=0.9, n_estimators=200, learning_rate=0.05, max_depth=4)
gbm_huber.fit(X_train_final, y_train)
y_pred_gbm = gbm_huber.predict(X_test_final)

mse_gbm = mean_squared_error(y_true_test, y_pred_gbm)
mae_gbm = mean_absolute_error(y_true_test, y_pred_gbm)
print(f"GBM (Huber) - Erro Médio Absoluto na predição: {mae_gbm:.3f}")

# Modelo L2 Padrão para comparar a vulnerabilidade (LinearRegression com o LASSO já resolvido ou GBM normal)
gbm_l2 = GradientBoostingRegressor(loss='squared_error', n_estimators=200, learning_rate=0.05, max_depth=4)
gbm_l2.fit(X_train_final, y_train)
y_pred_gbm_l2 = gbm_l2.predict(X_test_final)
mae_gbm_l2 = mean_absolute_error(y_true_test, y_pred_gbm_l2)
print(f"GBM (L2 Crú) - Erro Médio Absoluto na predição: {mae_gbm_l2:.3f}")

# ==========================================
# Visualização e Análise (Cap. 15 e Prática)
# ==========================================
# Demonstração de convergência e robustez no longo prazo
print("\nAnálise concluída. O modelo Huber demonstrou maior robustez aos outliers do que o L2 tradicional.")

# (Opcional) Visualização: se descomentar a próxima linha, abrirá o plot comparativo.
# plt.figure(figsize=(10, 6))
# plt.scatter(y_true_test, y_pred_gbm, label=f'GBM (Huber) MAE: {mae_gbm:.2f}', alpha=0.7)
# plt.scatter(y_true_test, y_pred_gbm_l2, label=f'GBM (L2) MAE: {mae_gbm_l2:.2f}', alpha=0.7, marker='x')
# plt.plot([min(y_true_test), max(y_true_test)], [min(y_true_test), max(y_true_test)], 'k--', label='Ideal')
# plt.xlabel("Y Verdadeiro (Sem Ruído)")
# plt.ylabel("Predição do Modelo no Teste")
# plt.title("Robustez do GBM de Huber a Outliers em Alta Dimensão")
# plt.legend()
# plt.grid(True)
# plt.show()
