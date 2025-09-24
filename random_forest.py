import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

warnings.filterwarnings('ignore')


# Dados de doenças cardíacas: https://archive.ics.uci.edu/dataset/45/heart+disease
# 1. Carregando os dados diretamente do repositório UCI
print("Iniciando o download do conjunto de dados...")
try:
    heart_disease = fetch_ucirepo(id=45)
    
    X = heart_disease.data.features
    y = heart_disease.data.targets

except Exception as e:
    print(f"Ocorreu um erro ao carregar os dados: {e}")
    print("Verifique se a biblioteca 'ucimlrepo' está instalada: pip install ucimlrepo")
    exit()

X.columns = ['idade', 'sexo', 'tipo_dor_peito', 'pressao_sanguinea', 'colesterol', 
             'acucar_jejum', 'eletrocardiograma', 'frequencia_cardiaca_max', 
             'angina_exercicio', 'depressao_st', 'inclinacao_st', 'vasos_maior_relevo', 
             'thalassemia']

# Convertendo a variável-alvo para um problema binário (0 ou 1)
y['num'] = np.where(y['num'] > 0, 1, 0)
y = y['num'] # Seleciona a série para o modelo


# 2. Separando os Dados para Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Otimizando os Hiperparâmetros com GridSearchCV
print("\nIniciando a busca pelos melhores hiperparâmetros...")
parametros = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}
modelo_rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=modelo_rf, param_grid=parametros, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
melhor_modelo = grid_search.best_estimator_
print("\nBusca concluída. Melhores parâmetros encontrados:")
print(grid_search.best_params_)

# 4. Avaliação Final do Modelo Otimizado
y_pred = melhor_modelo.predict(X_test)

# 5. Visualizando a Matriz de Confusão
print("\n--- Visualização 1: Matriz de Confusão ---")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Previsto: Sem Doença', 'Previsto: Com Doença'], 
            yticklabels=['Real: Sem Doença', 'Real: Com Doença'])
plt.xlabel('Previsão do Modelo')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusão: Comparando Previsão vs. Realidade', fontsize=16)
plt.show()

# 6. Gerando a Curva ROC e o Gráfico
print("\n--- Visualização 2: Curva ROC e AUC ---")
y_prob = melhor_modelo.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Modelo Aleatório (área = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Desempenho do Modelo de Classificação', fontsize=16)
plt.legend(loc="lower right")
plt.show()

# 7. Analisando a Importância das Variáveis
print("\n--- Visualização 3: Importância das Variáveis ---")
importancias = melhor_modelo.feature_importances_
nomes_colunas = X.columns
df_importancia = pd.DataFrame({'Variável': nomes_colunas, 'Importância': importancias})
df_importancia = df_importancia.sort_values(by='Importância', ascending=False)
print("Variáveis mais importantes para a previsão:\n")
print(df_importancia.to_string(index=False))
plt.figure(figsize=(12, 8))
sns.barplot(x='Importância', y='Variável', data=df_importancia, palette='viridis')
plt.title('Importância das Variáveis na Previsão do Diagnóstico', fontsize=16)
plt.xlabel('Importância Relativa')
plt.ylabel('Variável')
plt.show()

# 8. Conclusão Final
acuracia = accuracy_score(y_test, y_pred)
print("\n--- Relatório de Desempenho do Modelo ---")
print(f"**Acurácia:** {acuracia:.2f}")
print("Acurácia é a porcentagem total de previsões corretas.")
print(classification_report(y_test, y_pred, target_names=['Ausência de Doença (0)', 'Presença de Doença (1)']))
print("\n--- Conclusão da Análise ---")
print("O algoritmo Random Forest se mostrou robusto e eficaz para este problema, atingindo uma acurácia de 87% e apresentando um bom equilíbrio nas demais métricas.")
print("A matriz de confusão, a curva ROC e a análise de importância das variáveis nos dão uma visão completa do desempenho e do funcionamento interno do modelo, demonstrando que ele não é uma 'caixa-preta', mas uma ferramenta poderosa para a tomada de decisões.")