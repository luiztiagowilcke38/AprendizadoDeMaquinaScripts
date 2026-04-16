from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Matriz Binaria: Linha=Transacao, Coluna=Produto
data = pd.DataFrame({
    'pao': [1, 1, 0, 1],
    'leite': [1, 0, 1, 1],
    'cafe': [0, 1, 1, 1]
})

frequentes = apriori(data, min_support=0.5, use_colnames=True)
regras = association_rules(frequentes, metric="confidence", min_threshold=0.7)

# No ESL, discute-se como regras de associacao sao um caso 
# de densidade conjunta em variaveis binarias esparsas.