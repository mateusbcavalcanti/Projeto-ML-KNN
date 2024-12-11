import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


# base = pd.read_excel('C:\\Users\\Notebook\\OneDrive\\Documents\\Faculdade\\Nono Período\\Aprendizado de máquina\\Atividades\\Projeto ML AV2\\Projeto-ML-KNN\\dadosRating_crip.xlsx')
# # excluindo colunas equivalentes a outras
# base = base.drop(['RAZÃO SOCIAL','PROCESSO ADMINISTRATIVO'] ,axis=1)
# # print(len(base))
# #excluindo instancias nulas

# base = base.dropna(subset=['CNPJCPF'])
# base = base.dropna()
# # print(len(base))
# #definindo a feature
# X = base.drop('RATING', axis=1)

# #definindo a label
# y = base['RATING']
treino = pd.read_csv('C:\\Users\\Notebook\\OneDrive\\Documents\\Faculdade\\Nono Período\\Aprendizado de máquina\\Atividades\\Projeto ML AV2\\Projeto-ML-KNN\\treino.csv', sep=',')
teste = pd.read_csv('C:\\Users\\Notebook\\OneDrive\\Documents\\Faculdade\\Nono Período\\Aprendizado de máquina\\Atividades\\Projeto ML AV2\\Projeto-ML-KNN\\teste.csv', sep=',')
validacao = pd.read_csv('C:\\Users\\Notebook\\OneDrive\\Documents\\Faculdade\\Nono Período\\Aprendizado de máquina\\Atividades\\Projeto ML AV2\\Projeto-ML-KNN\\validacao.csv', sep=',')
#normalizando os dados
scaler = MinMaxScaler()

# dividindo a base em treino(70%) teste(15%) e validacao(15%)
# separando em treino teste e validacao
X_train = treino.drop(columns=['RATING'])
y_train = treino['RATING']

X_test = teste.drop(columns=['RATING'])
y_test = teste['RATING']

X_val = validacao.drop(columns=['RATING'])
y_val = validacao['RATING']


X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(scaler.fit_transform(X_test),columns = X_test.columns)
X_val = pd.DataFrame(scaler.fit_transform(X_val), columns = X_val.columns)
#testando o uso de random forest
# clf = RandomForestClassifier(n_estimators=1000)
# clf.fit(X_train, y_train)
# print(f"Precisao usando o RF {clf.score(X_test, y_test)}")
# y_preds = clf.predict(X_test)
# print(np.mean(y_preds == y_test))

#usando o KNN
k=5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
#mostra o quão preciso esta a previsao dos dados
print(f"Precisao usando o KNN {knn.score(X_test, y_test)}")
#comparando a previsao com o valor original já classificado
y_preds = knn.predict(X_test)
#a taxa recebida na comparacao eh a mesma do score
print(np.mean(y_preds == y_test))


