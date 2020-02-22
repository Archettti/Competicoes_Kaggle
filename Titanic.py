O objetivo deste desafio de Data Science é utilizar os dados disponíveis para medir a probabilidade de sobrevivência dos passageiros do Titanic.

# Bibliotecas para carregar/manipular os dados
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import seaborn as sns
import re as re

# Biblioteca para ML
from sklearn import ensemble
from sklearn.metrics import classification_report, confusion_matrix

# Carrega a base de dados de treino e teste
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")


-- INÍCIO DA ANÁLISE EXPLORATÓRIA --

# verificando as dimensões dos Datasets
print("Variáveis:\t{}\nEntradas:\t{}".format(train.shape[1], train.shape[0]))
print("Variáveis:\t{}\nEntradas:\t{}".format(test.shape[1], test.shape[0]))

# Identificar o tipo de dado e os 5 primeiros exemples do dataset
display(train.dtypes)
display(train.head())

# Percentual de missing value
(train.isnull().sum() / train.shape[0]).sort_values(ascending=False)*100
#(test.isnull().sum() / test.shape[0]).sort_values(ascending=False)*100

# Visão gerão da distribuição de cada variável
train.describe()

# Ver histograma das variáveis numéricas
train.hist(figsize=(10,8));

# Analisar a probabilidade de sobrevivência pelo Sexo
train[['Sex', 'Survived']].groupby(['Sex']).mean()

# Plotar os gráficos de barra para Survived vs. Sex, Pclass e Embarked
fig, (axis1, axis2, axis3) = plt.subplots(1,3, figsize=(12,4))

sns.barplot(x='Sex', y='Survived', data=train, ax=axis1)
sns.barplot(x='Pclass', y='Survived', data=train, ax=axis2)
sns.barplot(x='Embarked', y='Survived', data=train, ax=axis3);

# Influência da idade na probabilidade de sobrevivência. 
# É possível identificar um índice alto de sobrevivência entre crianças
age_survived = sns.FacetGrid(train, col='Survived')
age_survived.map(sns.distplot, 'Age');

-- JUNTAR DATASET TREINO E TESTE E TRATAR MISSING VALUES --

# Juntar os datasets de treino e teste
# salvar os índices dos datasets para recuperação posterior
train_idx = train.shape[0]
test_idx = test.shape[0]

# salvar PassengerId para submissao ao Kaggle
passengerId = test['PassengerId']

# extrair coluna 'Survived' e excluí-la do dataset treino
label = train.Survived.copy()
train.drop(['Survived'], axis=1, inplace=True)

# concatenar treino e teste em um único DataFrame
df_merged = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

# antes
print("train.shape: ({} x {})".format(train.shape[0], train.shape[1]))
print("test.shape: ({} x {})".format(test.shape[0], test.shape[1]))

# e depois da concatenação
print("df_merged.shape: ({} x {})".format(df_merged.shape[0], df_merged.shape[1]))

# Desconsiderar features a princípio não relevantes
df_merged.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
#print("df_merged.shape: ({} x {})".format(df_merged.shape[0], df_merged.shape[1]))

# Ver missing values
df_merged.isnull().sum()

# recuperar os titulos dos nomes dos passageiros
def extrai_titulo(df):
    df['Titulo'] = df.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip().lower())
    
extrai_titulo(df_merged)
df_merged.head()

# Agora podemos retirar a coluna Name
df_merged.drop(['Name'], axis=1, inplace=True)
df_merged.head()

# verificando a distribuição dos titulos
fig = plt.figure(figsize=(11,6))
fig = df_merged['Titulo'].value_counts().plot.barh()
fig.set_title('Abreviacao')
fig.set_ylabel('Nomes dos titulos')
fig.set_xlabel('Quantidade por titulo')

# age - mediana
age_median = df_merged['Age'].median()
df_merged['Age'].fillna(age_median, inplace=True)

# fare - mediana
fare_median = df_merged['Fare'].median()
df_merged['Fare'].fillna(fare_median, inplace=True)

# embarked - moda
embarked_top = df_merged['Embarked'].value_counts()[0]
df_merged['Embarked'].fillna(embarked_top, inplace=True)

# Ver missing values após tratamento
df_merged.isnull().sum()

-- PREPARAR AS VARIÁVEIS CATEGÓRICAS PARA O MODELO --

# transformar os dados de entrada que estão em formato categoria para números
# converter 'Sex' em 0 e 1
df_merged['Sex'] = df_merged['Sex'].map({'male': 0, 'female': 1})

# dummie variables para 'Embarked'
embarked_dummies = pd.get_dummies(df_merged['Embarked'], prefix='Embarked')
df_merged = pd.concat([df_merged, embarked_dummies], axis=1)
df_merged.drop('Embarked', axis=1, inplace=True)
df_merged.drop(['Embarked_914'], axis=1, inplace=True)

# tratamento dos titulos raros
def titulo_raro(passageiro):
    linha = passageiro
    if re.search('mrs', linha):
        return 'mrs'
    elif re.search('mr', linha):
        return 'mr'
    elif re.search('miss',linha):
        return 'miss'
    elif re.search('master',linha):
        return 'master'
    else:
        return 'other'
    
# dummies para titulo
embarked_titulo = pd.get_dummies(df_merged['Titulo'].apply(titulo_raro), prefix='Titulo')
df_merged = pd.concat([df_merged, embarked_titulo], axis=1)
df_merged.drop('Titulo', axis=1, inplace=True)
df_merged.sample(5)

-- DIVIDIR O DATASET CONCATENADO E TRATADO EM TREINO E TESTE --

# recuperar datasets de treino e teste
train = df_merged.iloc[:train_idx]
test = df_merged.iloc[train_idx:]

display(train.head())
display(test.head())
display(label.head())

-- IMPLEMENTAR O MODELO --

# criar um modelo de árvore de decisão
model = ensemble.GradientBoostingClassifier(n_estimators=500,
                                    validation_fraction=0.2,
                                    n_iter_no_change=5, tol=0.01,
                                    random_state=0)
model.fit(train, label)

# verificar a acurácia do modelo
acc_model = round(model.score(train, label) * 100, 2)
print("Acurácia do modelo de Árvore de Decisão: {}".format(acc_model))

-- GERAR ARQUIVO PARA SUBMISSÃO --

predict = model2.predict(test)

# gerar arquivo csv
submission = pd.DataFrame({
    "PassengerId": passengerId,
    "Survived": predict
})

submission.to_csv('./my_submission.csv', index=False)
print("Arquivo gerado com sucesso")
submission
