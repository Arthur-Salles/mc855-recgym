import pandas as pd
import numpy as np
from pyts.transformation import BagOfPatterns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


import scipy.sparse as sp

import numpy as np
import pyts.transformation.bag_of_patterns as bop_module

# ---- Correção Universal do BagOfPatterns para SciPy>=1.15 ----  //chatgpt
# Refaz a função fit_transform() para evitar chamar o atributo .A, que foi removido
def _fit_transform_corrigido(self, X, y=None):
    # Ajusta (fit) e depois transforma (transform) (seguro para todas as versões do pyts)
    self.fit(X, y)
    X_bop = self.transform(X)
    # Se o resultado for esparso (sparse), converte para array denso
    if sp.issparse(X_bop):
        X_bop = X_bop.toarray()
    # Garante que é um array numpy
    return np.asarray(X_bop)

# "Injeta" a nossa função corrigida na classe original do BagOfPatterns
bop_module.BagOfPatterns.fit_transform = _fit_transform_corrigido


# ---------- CSVs ----------
treino = pd.read_csv("train_data.csv")
validacao = pd.read_csv("validation_data.csv")
teste = pd.read_csv("test_data.csv")

# ---------- Colunas dos sensores ----------
colunas_sensores = ['A_x', 'A_y', 'A_z', 'G_x', 'G_y', 'G_z', 'C_1']

# ---------- Calcula o comprimento das series temporais ----------
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

# ---------- Remodela os datasets p/ ->(amostras, comprimento_alvo * n_sensores) ----------
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
X_validacao   = escalador.transform(X_validacao)
X_teste  = escalador.transform(X_teste)

# ---------- Bag of Patterns (BoP) ----------
tamanho_janela = max(10, comprimento_alvo // 30) #chutei 30 e no min 10
bop = BagOfPatterns(window_size=tamanho_janela, word_size=6, n_bins=4, sparse=True) 

bop.fit(X_treino)

# Transforma os dados p/ o espaço de vetores BoP
# Transforma e converte manualmente p/ arrays densos, pq sao de tamanhos diferentes
X_treino_bop = bop.transform(X_treino)
if sp.issparse(X_treino_bop):
    X_treino_bop = X_treino_bop.toarray()

X_validacao_bop = bop.transform(X_validacao)
if sp.issparse(X_validacao_bop):
    X_validacao_bop = X_validacao_bop.toarray()

X_teste_bop = bop.transform(X_teste)
if sp.issparse(X_teste_bop):
    X_teste_bop = X_teste_bop.toarray()


print("BoP feature space:", X_treino_bop.shape)

def mostra_matriz_confusao(y_true, y_pred, titulo):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', colorbar=True)
    plt.title(titulo)
    #plt.show()

# ------------------ Random Forest ------------------
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_treino_bop, y_treino)

# Validação
y_pred_validacao = clf.predict(X_validacao_bop)
acc_validacao = accuracy_score(y_validacao, y_pred_validacao)
f1_validacao = f1_score(y_validacao, y_pred_validacao, average='weighted')
print(f"Validation accuracy (Random Forest): {acc_validacao:.3f}")
print(f"Validation F1-score (Random Forest): {f1_validacao:.3f}")
# mostra_matriz_confusao(y_validacao, y_pred_validacao, "Matriz de Confusão - Validação (Random Forest)")

# Teste
y_pred_teste = clf.predict(X_teste_bop)
acc_teste = accuracy_score(y_teste, y_pred_teste)
f1_teste = f1_score(y_teste, y_pred_teste, average='weighted')
print(f"Test accuracy (Random Forest): {acc_teste:.3f}")
print(f"Test F1-score (Random Forest): {f1_teste:.3f}")
# mostra_matriz_confusao(y_teste, y_pred_teste, "Matriz de Confusão - Teste (Random Forest)")

# ------------------ Logistic Regression ------------------
clf_logreg = LogisticRegression(max_iter=5000, solver='saga', random_state=42)
clf_logreg.fit(X_treino_bop, y_treino)

# Validação
y_pred_validacao_logreg = clf_logreg.predict(X_validacao_bop)
acc_validacao_logreg = accuracy_score(y_validacao, y_pred_validacao_logreg)
f1_validacao_logreg = f1_score(y_validacao, y_pred_validacao_logreg, average='weighted') 
print(f"Validation accuracy (LogReg): {acc_validacao_logreg:.3f}")
print(f"Validation F1-score (LogReg): {f1_validacao_logreg:.3f}") 
# mostra_matriz_confusao(y_validacao, y_pred_validacao_logreg, "Matriz de Confusão - Validação (Logistic Regression)")
# Teste
y_pred_teste_logreg = clf_logreg.predict(X_teste_bop)
acc_teste_logreg = accuracy_score(y_teste, y_pred_teste_logreg)
f1_teste_logreg = f1_score(y_teste, y_pred_teste_logreg, average='weighted') 
print(f"Test accuracy (LogReg): {acc_teste_logreg:.3f}")
print(f"Test F1-score (LogReg): {f1_teste_logreg:.3f}") 
# mostra_matriz_confusao(y_validacao, y_pred_validacao_logreg, "Matriz de Confusão - Validação (Logistic Regression)")

# ------------------ SVM ------------------
clf_svm = SVC(kernel='rbf', C=1.0, random_state=42)
clf_svm.fit(X_treino_bop, y_treino)

# Validação
y_pred_validacao_svm = clf_svm.predict(X_validacao_bop)
acc_validacao_svm = accuracy_score(y_validacao, y_pred_validacao_svm)
f1_validacao_svm = f1_score(y_validacao, y_pred_validacao_svm, average='weighted') 
print(f"Validation accuracy (SVM): {acc_validacao_svm:.3f}")
print(f"Validation F1-score (SVM): {f1_validacao_svm:.3f}") 
# mostra_matriz_confusao(y_validacao, y_pred_validacao_svm, "Matriz de Confusão - Validação (SVM)")

# Teste
y_pred_teste_svm = clf_svm.predict(X_teste_bop)
acc_teste_svm = accuracy_score(y_teste, y_pred_teste_svm)
f1_teste_svm = f1_score(y_teste, y_pred_teste_svm, average='weighted') 
print(f"Test accuracy (SVM): {acc_teste_svm:.3f}")
print(f"Test F1-score (SVM): {f1_teste_svm:.3f}") 
# mostra_matriz_confusao(y_teste, y_pred_teste_svm, "Matriz de Confusão - Teste (SVM)")

# ------------------ K-Nearest Neighbors ------------------
clf_knn = KNeighborsClassifier(n_neighbors=5)  
clf_knn.fit(X_treino_bop, y_treino)

# Validação
y_pred_validacao_knn = clf_knn.predict(X_validacao_bop)
acc_validacao_knn = accuracy_score(y_validacao, y_pred_validacao_knn)
f1_validacao_knn = f1_score(y_validacao, y_pred_validacao_knn, average='weighted') 
print(f"Validation accuracy (KNN): {acc_validacao_knn:.3f}")
print(f"Validation F1-score (KNN): {f1_validacao_knn:.3f}") 
mostra_matriz_confusao(y_validacao, y_pred_validacao_knn, "Matriz de Confusão - Validação (KNN)")

# Teste
y_pred_teste_knn = clf_knn.predict(X_teste_bop)
acc_teste_knn = accuracy_score(y_teste, y_pred_teste_knn)
f1_teste_knn = f1_score(y_teste, y_pred_teste_knn, average='weighted') 
print(f"Test accuracy (KNN): {acc_teste_knn:.3f}")
print(f"Test F1-score (KNN): {f1_teste_knn:.3f}") 
mostra_matriz_confusao(y_teste, y_pred_teste_knn, "Matriz de Confusão - Teste (KNN)")

plt.show()