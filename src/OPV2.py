import pandas as pd
import numpy as np
import ordpy as ord  
import math          
import itertools     
from tqdm import tqdm 
from sklearn.preprocessing import MinMaxScaler  
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score  
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ---------- CSVs ----------
try:
    treino = pd.read_csv("train_data.csv")
    validacao = pd.read_csv("validation_data.csv")
    teste = pd.read_csv("test_data.csv")
except FileNotFoundError:
    print("Erro: Arquivos train_data.csv, validation_data.csv, ou test_data.csv não encontrados.")
    exit()


# ---------- Colunas dos sensores ----------
colunas_sensores = ['A_x', 'A_y', 'A_z', 'G_x', 'G_y', 'G_z', 'C_1']

# ---------- Calcula o comprimento das séries (Apenas para informação) ----------
def calcula_comprimento_alvo(*dfs):
    comprimentos = []
    for df in dfs:
        for _, grupo in df.groupby("sample"):
            comprimentos.append(len(grupo))
    print("Comprimentos -> min:", min(comprimentos), 
          "median:", np.median(comprimentos), 
          "max:", max(comprimentos))

print("--- Estatísticas do Comprimento das Séries (Original) ---")
calcula_comprimento_alvo(treino, validacao, teste)

def remodela_series_temporais(df):
    X_remodelado = []
    y_rotulos = []
    
    for sample_id, grupo in df.groupby("sample"):
        serie_2d = grupo[colunas_sensores].values 
        
        serie_final = serie_2d.T 
        
        X_remodelado.append(serie_final)
        y_rotulos.append(grupo["label"].iloc[0])
        
    return X_remodelado, np.array(y_rotulos)

# --- Criando os datasets ---
print("\n--- Carregando dados (sem remodelamento) ---")
X_treino_list, y_treino = remodela_series_temporais(treino)
X_validacao_list, y_validacao = remodela_series_temporais(validacao)
X_teste_list, y_teste = remodela_series_temporais(teste)

print(f"Número de amostras de treino: {len(X_treino_list)}")
print(f"Forma da primeira amostra de treino: {X_treino_list[0].shape}")
print(f"Forma da segunda amostra de treino: {X_treino_list[1].shape}") # Prova de comprimentos variados


# ---------- Funções de Padrões Ordinais  ----------

def get_op_from_ts(x, dx, dy=1, taux=10):
    try:
        symbols = ord.ordinal_sequence(x, dx=dx, dy=dy, taux=taux)

        if np.shape(symbols).__len__() == 3:
            symbols = symbols.reshape(-1, dx*dy) 
        
        all_symbols = np.asarray(list(itertools.permutations(range(dx*dy))), dtype='int')
        symbols, symbols_count = np.unique(symbols, return_counts=True, axis=0)
        probabilities = symbols_count / symbols_count.sum()

        all_symbols_str = np.apply_along_axis(np.char.strip, 0, 
                                              np.apply_along_axis(np.array2string, 1, all_symbols, separator=''),
                                              chars="[]")

        if len(probabilities) == math.factorial(dx*dy):
            return dict(zip(all_symbols_str, probabilities))
        
        all_probs = np.full(math.factorial(dx*dy), 0.)
        dict_probs = dict(zip(all_symbols_str, all_probs))

        # Converte 'symbols' de volta para string para corresponder às chaves
        symbols_str = np.apply_along_axis(np.char.strip, 0,
                                          np.apply_along_axis(np.array2string, 1, symbols, separator=''),
                                          chars="[]")

        for symbol, probability in zip(symbols_str, probabilities):
            if symbol in dict_probs:
                dict_probs[symbol] = probability
            
        return dict_probs

    except Exception as e:
        print(f"Erro em get_op_from_ts (amostra com forma {x.shape}): {e}")
        all_symbols = np.asarray(list(itertools.permutations(range(dx*dy))), dtype='int')
        all_symbols_str = np.apply_along_axis(np.char.strip, 0, 
                                              np.apply_along_axis(np.array2string, 1, all_symbols, separator=''),
                                              chars="[]")
        all_probs = np.full(math.factorial(dx*dy), 0.)
        return dict(zip(all_symbols_str, all_probs))


def get_op(datalist, dx, dy=1, taux=10):
    prob_vectors = []
    for x in datalist:
        dprob = get_op_from_ts(x, dx, dy, taux=taux)
        prob_vectors.append(list(dprob.values()))
    
    return np.array(prob_vectors)


# ---------- Aplica OP ----------
OP_DX = 3
OP_DY = 2
OP_TAUX = 10 

print(f"\n--- Extraindo Features de OP com dx={OP_DX}, dy={OP_DY}, taux={OP_TAUX} ---")
X_treino_op = get_op(X_treino_list, dx=OP_DX, dy=OP_DY, taux=OP_TAUX)
X_validacao_op = get_op(X_validacao_list, dx=OP_DX, dy=OP_DY, taux=OP_TAUX)
X_teste_op = get_op(X_teste_list, dx=OP_DX, dy=OP_DY, taux=OP_TAUX)

print("OP histogram feature space:", X_treino_op.shape)


# ---------- Normalizador ----------
print("Aplicando MinMaxScaler nas features de OP...")
escalador = MinMaxScaler()
X_treino_op = escalador.fit_transform(X_treino_op)
X_validacao_op = escalador.transform(X_validacao_op)
X_teste_op = escalador.transform(X_teste_op)

# ---------- Função de Matriz de Confusão ----------
def mostra_matriz_confusao(y_true, y_pred, titulo):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', colorbar=True)
    plt.title(titulo)

F1_AVG = 'binary' 

# ---------- Random Forest ----------
print("\n--- Treinando Random Forest ---")
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_treino_op, y_treino)

y_pred_validacao = clf.predict(X_validacao_op)
acc_validacao = accuracy_score(y_validacao, y_pred_validacao)
f1_validacao = f1_score(y_validacao, y_pred_validacao, average=F1_AVG, zero_division=0)
recall_validacao = recall_score(y_validacao, y_pred_validacao, average=F1_AVG, zero_division=0) # <--- ADICIONADO
print(f"Validation accuracy (Random Forest): {acc_validacao:.3f}")
print(f"Validation F1-score  (Random Forest): {f1_validacao:.3f}") 
print(f"Validation Recall (Random Forest): {recall_validacao:.3f}") # <--- ADICIONADO

y_pred_teste = clf.predict(X_teste_op)
acc_teste = accuracy_score(y_teste, y_pred_teste)
f1_teste = f1_score(y_teste, y_pred_teste, average=F1_AVG, zero_division=0)
recall_teste = recall_score(y_teste, y_pred_teste, average=F1_AVG, zero_division=0) # <--- ADICIONADO
print(f"Test accuracy (Random Forest): {acc_teste:.3f}")
print(f"Test F1-score  (Random Forest): {f1_teste:.3f}") 
print(f"Test Recall (Random Forest): {recall_teste:.3f}") # <--- ADICIONADO


# ---------- Logistic Regression  ----------
print("\n--- Treinando Logistic Regression ---")
clf_logreg = LogisticRegression(max_iter=5000, solver='saga', random_state=42)
clf_logreg.fit(X_treino_op, y_treino)

y_pred_validacao_logreg = clf_logreg.predict(X_validacao_op)
acc_validacao_logreg = accuracy_score(y_validacao, y_pred_validacao_logreg)
f1_validacao_logreg = f1_score(y_validacao, y_pred_validacao_logreg, average=F1_AVG, zero_division=0)
recall_validacao_logreg = recall_score(y_validacao, y_pred_validacao_logreg, average=F1_AVG, zero_division=0) # <--- ADICIONADO
print(f"Validation accuracy (LogReg): {acc_validacao_logreg:.3f}")
print(f"Validation F1-score  (LogReg): {f1_validacao_logreg:.3f}") 
print(f"Validation Recall (LogReg): {recall_validacao_logreg:.3f}") # <--- ADICIONADO

y_pred_teste_logreg = clf_logreg.predict(X_teste_op)
acc_teste_logreg = accuracy_score(y_teste, y_pred_teste_logreg)
f1_teste_logreg = f1_score(y_teste, y_pred_teste_logreg, average=F1_AVG, zero_division=0)
recall_teste_logreg = recall_score(y_teste, y_pred_teste_logreg, average=F1_AVG, zero_division=0) # <--- ADICIONADO
print(f"Test accuracy (LogReg): {acc_teste_logreg:.3f}")
print(f"Test F1-score  (LogReg): {f1_teste_logreg:.3f}") 
print(f"Test Recall (LogReg): {recall_teste_logreg:.3f}") # <--- ADICIONADO


# ---------- SVM  ----------
print("\n--- Treinando SVM ---")
clf_svm = SVC(kernel='rbf', C=1.0, random_state=42)
clf_svm.fit(X_treino_op, y_treino)

y_pred_validacao_svm = clf_svm.predict(X_validacao_op)
acc_validacao_svm = accuracy_score(y_validacao, y_pred_validacao_svm)
f1_validacao_svm = f1_score(y_validacao, y_pred_validacao_svm, average=F1_AVG, zero_division=0)
recall_validacao_svm = recall_score(y_validacao, y_pred_validacao_svm, average=F1_AVG, zero_division=0) # <--- ADICIONADO
print(f"Validation accuracy (SVM): {acc_validacao_svm:.3f}")
print(f"Validation F1-score  (SVM): {f1_validacao_svm:.3f}") 
print(f"Validation Recall (SVM): {recall_validacao_svm:.3f}") # <--- ADICIONADO

y_pred_teste_svm = clf_svm.predict(X_teste_op)
acc_teste_svm = accuracy_score(y_teste, y_pred_teste_svm)
f1_teste_svm = f1_score(y_teste, y_pred_teste_svm, average=F1_AVG, zero_division=0)
recall_teste_svm = recall_score(y_teste, y_pred_teste_svm, average=F1_AVG, zero_division=0) # <--- ADICIONADO
print(f"Test accuracy (SVM): {acc_teste_svm:.3f}")
print(f"Test F1-score  (SVM): {f1_teste_svm:.3f}")
print(f"Test Recall (SVM): {recall_teste_svm:.3f}") # <--- ADICIONADO

# ---------- K-Nearest Neighbors  ----------
print("\n--- Treinando K-Nearest Neighbors ---")
clf_knn = KNeighborsClassifier(n_neighbors=5) 
clf_knn.fit(X_treino_op, y_treino)

y_pred_validacao_knn = clf_knn.predict(X_validacao_op)
acc_validacao_knn = accuracy_score(y_validacao, y_pred_validacao_knn)
f1_validacao_knn = f1_score(y_validacao, y_pred_validacao_knn, average=F1_AVG, zero_division=0)
recall_validacao_knn = recall_score(y_validacao, y_pred_validacao_knn, average=F1_AVG, zero_division=0) # <--- ADICIONADO
print(f"Validation accuracy (KNN): {acc_validacao_knn:.3f}")
print(f"Validation F1-score  (KNN): {f1_validacao_knn:.3f}") 
print(f"Validation Recall (KNN): {recall_validacao_knn:.3f}") # <--- ADICIONADO

y_pred_teste_knn = clf_knn.predict(X_teste_op)
acc_teste_knn = accuracy_score(y_teste, y_pred_teste_knn)
f1_teste_knn = f1_score(y_teste, y_pred_teste_knn, average=F1_AVG, zero_division=0)
recall_teste_knn = recall_score(y_teste, y_pred_teste_knn, average=F1_AVG, zero_division=0) # <--- ADICIONADO
print(f"Test accuracy (KNN): {acc_teste_knn:.3f}")
print(f"Test F1-score  (KNN): {f1_teste_knn:.3f}")
print(f"Test Recall (KNN): {recall_teste_knn:.3f}") # <--- ADICIONADO


# ---------- Plot de Entropia  ----------
print(f"\n--- Gerando Gráfico de Complexidade-Entropia (C-H) ---")

params_ce = {'dx': OP_DX, 'dy': OP_DY, 'taux': OP_TAUX}

train_ce = []
print("Calculando C-H para amostras de treino...")
for dt in tqdm(X_treino_list):
    train_ce.append(ord.complexity_entropy(dt, **params_ce))
train_ce = np.array(train_ce)

colors = ['tab:green' if k == 1 else 'tab:red' for k in y_treino]

# Curvas teóricas
hc_max_curve = ord.maximum_complexity_entropy(dx=params_ce['dx'], dy=params_ce['dy']).T
hc_min_curve = ord.minimum_complexity_entropy(dx=params_ce['dx'], dy=params_ce['dy']).T

hmin, cmin = hc_min_curve
hmax, cmax = hc_max_curve

print("Exibindo gráficos...")
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(hmin, cmin, linewidth=1.5, color="#161414", zorder=0, label="Limite Mínimo")
ax.plot(hmax, cmax, linewidth=1.5, color="#161414", zorder=0, label="Limite Máximo")
ax.scatter(train_ce.T[0,:], train_ce.T[1,:], c=colors, alpha=0.7)
ax.set_xlabel("Entropia (Normalizada)")
ax.set_ylabel("Complexidade")
ax.set_title(f"Plano C-H (dx={params_ce['dx']}, dy={params_ce['dy']}, taux={params_ce['taux']})")



plt.show()