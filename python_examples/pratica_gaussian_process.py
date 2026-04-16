import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# ==========================================
# Prática: Processos Gaussianos (GP) e Incerteza
# Baseado na Teoria do Livro: Cap. 06 (Métodos de Kernel)
# ==========================================
# "Diferente da SVM, os GPs são modelos Bayesianos... A predição em pontos novos 
# fornece não apenas o valor, mas a variância associada (barra de erro)."

np.random.seed(1)

# Gerando dados de treinamento: Função alvo f(x) = x * sin(x)
def f(x):
    return x * np.sin(x)

# Selecionamos alguns pontos ruidosos para treinar o modelo
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
y = f(X).ravel() + np.random.normal(0, 0.5, X.shape[0])

# Criamos pontos de teste para avaliar a predição ao longo de um domínio contínuo
x_test = np.atleast_2d(np.linspace(0, 10, 1000)).T

# Instanciando o Kernel (Kernel Constante multiplicado por RBF)
# O RBF mapeia para dimensão infinita de Hilbert e a variância dos dados diz a incerteza
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

# Ajustando o Processo Gaussiano
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=0.5)
gp.fit(X, y)

# Predição: y_pred é a média; sigma é o desvio padrão (incerteza)
y_pred, sigma = gp.predict(x_test, return_std=True)

print("Ajuste do Processo Gaussiano concluído.")
print(f"Kernel final otimizado: {gp.kernel_}")
print("Para visualizar o gráfico com o Intervalo de Confiança (Barra de Erro), garanta que está rodando em um ambiente com interface gráfica.")

# Visualizando os resultados que corroboram com o Cap. 06
plt.figure(figsize=(10, 5))
plt.plot(x_test, f(x_test), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=12, label='Observações (Treinamento)')
plt.plot(x_test, y_pred, 'b-', label='Predição do Processo Gaussiano')
plt.fill_between(x_test[:, 0], y_pred - 1.9600 * sigma, y_pred + 1.9600 * sigma,
                 alpha=0.2, color='b', label='Incerteza/Confiança (95%)')
plt.xlabel('Variável de Entrada (x)')
plt.ylabel('Variável Alvo (y)')
plt.title('Processos Gaussianos: Estimativa Pontual e Intervalo de Incerteza (Cap. 06)')
plt.legend(loc='upper left')
plt.grid(True)
# plt.show() # Descomente para exibir a janela do gráfico localmente
