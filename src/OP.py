import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score 
from itertools import permutations
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ---------- CSVs ----------
treino = pd.read_csv("train_data.csv")
validacao = pd.read_csv("validation_data.csv")
teste = pd.read_csv("test_data.csv")

# ---------- Colunas dos sensores ----------
colunas_sensores = ['A_x', 'A_y', 'A_z', 'G_x', 'G_y', 'G_z', 'C_1']

# ---------- Calcula o comprimento das séries ----------
def calcula_comprimento_alvo(*dfs):
    comprimentos = []
    for df in dfs:
        for _, grupo in df.groupby("sample"):
            comprimentos.append(len(grupo))
    print("Comprimentos -> min:", min(comprimentos), 
          "median:", np.median(comprimentos), 
          "max:", max(comprimentos))
    return int(np.median(comprimentos))

comprimento_alvo = calcula_comprimento_alvo(treino, validacao, teste)

# ---------- Remodela os datasets ----------
def remodela_series_temporais(df, comprimento_alvo):
    X_remodelado = []
    y_rotulos = []
    
    for sample_id, grupo in df.groupby("sample"):
        serie = grupo[colunas_sensores].values.flatten()

        if len(serie) > comprimento_alvo:
            serie = serie[:comprimento_alvo]
        elif len(serie) < comprimento_alvo:
            tamanho_preenchimento = comprimento_alvo - len(serie)
            serie = np.pad(serie, (0, tamanho_preenchimento), mode="constant")

        X_remodelado.append(serie)
        y_rotulos.append(grupo["label"].iloc[0])
        
    return np.array(X_remodelado), np.array(y_rotulos)

X_treino, y_treino = remodela_series_temporais(treino, comprimento_alvo)
X_validacao, y_validacao = remodela_series_temporais(validacao, comprimento_alvo)
X_teste, y_teste = remodela_series_temporais(teste, comprimento_alvo)

print("Train shape:", X_treino.shape)
print("Val shape:", X_validacao.shape)
print("Test shape:", X_teste.shape)

# ---------- Normalizador ----------
escalador = StandardScaler()
X_treino = escalador.fit_transform(X_treino)
X_validacao = escalador.transform(X_validacao)
X_teste = escalador.transform(X_teste)

# ---------- Função segura de padrões ordinais ----------
def ordinal_patterns_hist(X, window_size=5):
    """
    Gera histograma de padrões ordinais para cada amostra.
    Cada amostra vira um vetor de frequências de todos os padrões possíveis.
    """
    
    n_samples, n_timestamps = X.shape
    
    all_patterns = list(permutations(range(window_size)))
    pattern_to_index = {p: i for i, p in enumerate(all_patterns)}
    
    n_patterns = len(all_patterns)
    
    result = np.zeros((n_samples, n_patterns))
    
    for i in range(n_samples):
        counts = Counter()
        for j in range(n_timestamps - window_size + 1):
            window = X[i, j:j+window_size]
            pattern = tuple(np.argsort(window))
            counts[pattern] += 1
        total = sum(counts.values())
        for p, c in counts.items():
            idx = pattern_to_index[p]
            result[i, idx] = c / total
    return result

# ---------- Aplica OP ----------
tamanho_janela = 5  

X_treino_op = ordinal_patterns_hist(X_treino, window_size=tamanho_janela)
X_validacao_op = ordinal_patterns_hist(X_validacao, window_size=tamanho_janela)
X_teste_op = ordinal_patterns_hist(X_teste, window_size=tamanho_janela)

print("OP histogram feature space:", X_treino_op.shape)

def mostra_matriz_confusao(y_true, y_pred, titulo):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', colorbar=True)
    plt.title(titulo)
    #plt.show()


# ---------- Random Forest ----------
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_treino_op, y_treino)

y_pred_validacao = clf.predict(X_validacao_op)
acc_validacao = accuracy_score(y_validacao, y_pred_validacao)
f1_validacao = f1_score(y_validacao, y_pred_validacao, average='weighted')
print(f"Validation accuracy (Random Forest): {acc_validacao:.3f}")
print(f"Validation F1-score (Random Forest): {f1_validacao:.3f}") 
# mostra_matriz_confusao(y_validacao, y_pred_validacao, "Matriz de Confusão - Validação (Random Forest)")


y_pred_teste = clf.predict(X_teste_op)
acc_teste = accuracy_score(y_teste, y_pred_teste)
f1_teste = f1_score(y_teste, y_pred_teste, average='weighted')
print(f"Test accuracy (Random Forest): {acc_teste:.3f}")
print(f"Test F1-score (Random Forest): {f1_teste:.3f}") 
# mostra_matriz_confusao(y_teste, y_pred_teste, "Matriz de Confusão - Teste (Random Forest)")


# ---------- Logistic Regression ----------
clf_logreg = LogisticRegression(max_iter=5000, solver='saga', random_state=42)
clf_logreg.fit(X_treino_op, y_treino)

y_pred_validacao_logreg = clf_logreg.predict(X_validacao_op)
acc_validacao_logreg = accuracy_score(y_validacao, y_pred_validacao_logreg)
f1_validacao_logreg = f1_score(y_validacao, y_pred_validacao_logreg, average='weighted')
print(f"Validation accuracy (LogReg): {acc_validacao_logreg:.3f}")
print(f"Validation F1-score (LogReg): {f1_validacao_logreg:.3f}") 
# mostra_matriz_confusao(y_validacao, y_pred_validacao_logreg, "Matriz de Confusão - Validação (Logistic Regression)")


y_pred_teste_logreg = clf_logreg.predict(X_teste_op)
acc_teste_logreg = accuracy_score(y_teste, y_pred_teste_logreg)
f1_teste_logreg = f1_score(y_teste, y_pred_teste_logreg, average='weighted')
print(f"Test accuracy (LogReg): {acc_teste_logreg:.3f}")
print(f"Test F1-score (LogReg): {f1_teste_logreg:.3f}") 
# mostra_matriz_confusao(y_validacao, y_pred_validacao_logreg, "Matriz de Confusão - Validação (Logistic Regression)")


# ---------- SVM ----------
clf_svm = SVC(kernel='rbf', C=1.0, random_state=42)
clf_svm.fit(X_treino_op, y_treino)

y_pred_validacao_svm = clf_svm.predict(X_validacao_op)
acc_validacao_svm = accuracy_score(y_validacao, y_pred_validacao_svm)
f1_validacao_svm = f1_score(y_validacao, y_pred_validacao_svm, average='weighted')
print(f"Validation accuracy (SVM): {acc_validacao_svm:.3f}")
print(f"Validation F1-score (SVM): {f1_validacao_svm:.3f}") 
# mostra_matriz_confusao(y_validacao, y_pred_validacao_svm, "Matriz de Confusão - Validação (SVM)")


y_pred_teste_svm = clf_svm.predict(X_teste_op)
acc_teste_svm = accuracy_score(y_teste, y_pred_teste_svm)
f1_teste_svm = f1_score(y_teste, y_pred_teste_svm, average='weighted')
print(f"Test accuracy (SVM): {acc_teste_svm:.3f}")
print(f"Test F1-score (SVM): {f1_teste_svm:.3f}") 
# mostra_matriz_confusao(y_teste, y_pred_teste_svm, "Matriz de Confusão - Teste (SVM)")


# ---------- K-Nearest Neighbors ----------
clf_knn = KNeighborsClassifier(n_neighbors=5) 
clf_knn.fit(X_treino_op, y_treino)

# Validação
y_pred_validacao_knn = clf_knn.predict(X_validacao_op)
acc_validacao_knn = accuracy_score(y_validacao, y_pred_validacao_knn)
f1_validacao_knn = f1_score(y_validacao, y_pred_validacao_knn, average='weighted')
print(f"Validation accuracy (KNN): {acc_validacao_knn:.3f}")
print(f"Validation F1-score (KNN): {f1_validacao_knn:.3f}") 
mostra_matriz_confusao(y_validacao, y_pred_validacao_knn, "Matriz de Confusão - Validação (KNN)")


# Teste
y_pred_teste_knn = clf_knn.predict(X_teste_op)
acc_teste_knn = accuracy_score(y_teste, y_pred_teste_knn)
f1_teste_knn = f1_score(y_teste, y_pred_teste_knn, average='weighted')
print(f"Test accuracy (KNN): {acc_teste_knn:.3f}")
print(f"Test F1-score (KNN): {f1_teste_knn:.3f}")
mostra_matriz_confusao(y_teste, y_pred_teste_knn, "Matriz de Confusão - Teste (KNN)")


plt.show()