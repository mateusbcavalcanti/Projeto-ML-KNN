import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

base = pd.read_excel('C:\\Users\\Notebook\\OneDrive\\Documents\\Faculdade\\Nono Período\\Aprendizado de máquina\\Atividades\\Projeto ML AV2\\dadosRating_crip.xlsx')
#excluindo colunas equivalentes a outras
base = base.drop(['RAZÃO SOCIAL','PROCESSO ADMINISTRATIVO'] ,axis=1)
print(len(base))
#excluindo instancias nulas

base = base.dropna(subset=['CNPJCPF'])
base = base.dropna()
print(len(base))
#definindo a feature
X = base.drop('RATING', axis=1)

#definindo a label
y = base['RATING']

#dividindo a base em treino(70%) teste(15%) e validacao(15%)
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.3)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)
#normalizando os dados

model = RandomForestRegressor()
model.fit(X_train, y_train)
model.score(X_test,y_test) 