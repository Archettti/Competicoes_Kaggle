# Bibliotecas para manipulação dos dados e arquivos

import numpy as np # algebra linear
import pandas as pd # processamento de dados, arquivos CSV I/O. 

# Para gráficos 
from matplotlib import pyplot as plt
import seaborn as sns

# Biblioteca para ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Ignorar avisos
import warnings  
warnings.filterwarnings('ignore')

# Verificando os arquivos disponibilizados no classroom (curso DM) importados para o diretório input
import os
os.listdir("../input/horse-colic-dataset")

# Carregando os datasets de treino e teste
train = pd.read_csv("../input/horse-colic-dataset/horse.csv")
test = pd.read_csv("../input/horse-colic-dataset/horseTest.csv")

# Verificando as dimensões dos datasets
print("--- Dataset Treino ---")
print("Variáveis:\t{}\nEntradas:\t{}\n".format(train.shape[1], train.shape[0]))

print("--- Dataset Teste ---")
print("Variáveis:\t{}\nEntradas:\t{}".format(test.shape[1], test.shape[0]))


# Identificando os tipos de dados e os exemplos de registro do dataset
display(train.dtypes)
display(train.dtypes.value_counts())
display(train.head())

#Visão geral das variáveis numéricas
train.describe()

# Checando e comparando os missing values das features de treino
qtde_nulos = train.isna().sum()

print(qtde_nulos)

plt.figure(figsize=(18,10))
plt.bar(range(len(qtde_nulos)), qtde_nulos)
plt.title('Missing Values x Features')
plt.xlabel('features')
plt.ylabel('missing')
plt.xticks(list(range(len(train.columns))), list(train.columns.values), rotation='vertical')
plt.show()

# Ver histograma das variáveis numéricas
train.hist(figsize=(18,15));

# Ver os rótulos do dataset e respectivas quantidades
sns.countplot(data=train, x='outcome');
print(train.outcome.value_counts())

# Analisar a probabilidade de sobrevivência caso o tratamento tenha sido cirúrgico
sns.countplot(data=train, x='outcome', hue='surgery');
plt.show()

sns.countplot(data=train, x='outcome', hue='pain');
plt.show()

g = sns.FacetGrid(data=train, col='outcome', margin_titles=True, height=6)
g.map(plt.hist, 'pulse')
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Outcome por Pulso');

g = sns.catplot(data=train, x='peripheral_pulse', col='outcome', kind='count');
g.fig.suptitle('Outcome por Pulso Periférico');
plt.subplots_adjust(top=0.85)

reduced_absent_pulse = train[train.outcome.isin(('died','euthanized')) & train.peripheral_pulse.isin(('reduced','absent'))]

g = sns.catplot(data=reduced_absent_pulse, x='capillary_refill_time', col='outcome', kind='count');
g.fig.suptitle('Outcome por Tempo de Preenchimento Capilar');
plt.subplots_adjust(top=0.85)

# Juntar os datasets de treino e teste para o tratamento dos dados em conjunto

# salvar os índices dos datasets para recuperação posterior
train_idx = train.shape[0]
test_idx = test.shape[0]

# concatenar treino e teste em um único DataFrame
df_merged = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

# antes
print("train.shape: ({} x {})".format(train.shape[0], train.shape[1]))
print("test.shape: ({} x {})\n".format(test.shape[0], test.shape[1]))

# e depois da concatenação
print("df_merged.shape: ({} x {})".format(df_merged.shape[0], df_merged.shape[1]))

for col in df_merged.columns.values:
    
    if (pd.isna(df_merged[col]).sum()) > 0: 
    
        if pd.isna(df_merged[col]).sum() > (50/100 * len(df_merged)): 
            print(col,"removido") 
            df_merged = df_merged.drop([col], axis=1) 
        
        elif (df_merged[col].dtype == 'object'):
            df_merged[col] = df_merged[col].fillna(df_merged[col].mode()[0])        
        
        else:
            df_merged[col] = df_merged[col].fillna(df_merged[col].median())
            
                
print(df_merged.shape)
print(df_merged.isna().sum())

# Aplicando o Label Encoder para o atributo "outcome"

df_merged["outcome"] = df_merged["outcome"].astype('category').cat.codes
df_merged.head()

df_merged_corr = df_merged.corr()
corr_values = df_merged_corr["outcome"].sort_values(ascending=False)
corr_values = abs(corr_values).sort_values(ascending=False)

print("Correlação das features numéricas com o resultado em ordem crescente")
print(abs(corr_values).sort_values(ascending=False))

# Removendo as features onde a correlação é praticamente inexistente 

df_merged = df_merged.drop(columns=['hospital_number'], axis=1)
df_merged = df_merged.drop(columns=['respiratory_rate'], axis=1)
df_merged = df_merged.drop(columns=['lesion_3'], axis=1)
df_merged = df_merged.drop(columns=['rectal_temp'], axis=1)

df_merged.head()

# Conversão de dados categóricos para numéricos - One Hot Encoding
df_merged = pd.get_dummies(df_merged)
df_merged.head(10)

# Recuperando datasets de treino e teste
train = df_merged.iloc[:train_idx]
test = df_merged.iloc[train_idx:]

# Verificando as dimensões dos Datasets após o tratamento dos dados
print("--- Dataset Treino ---")
print("Variáveis:\t{}\nEntradas:\t{}\n".format(train.shape[1], train.shape[0]))

print("--- Dataset Teste ---")
print("Variáveis:\t{}\nEntradas:\t{}".format(test.shape[1], test.shape[0]))

# Extraindo os resultados (outcome) e removendo dos datasets para o treinamento dos modelos
X_train = train.drop("outcome", axis=1).values
Y_train = train["outcome"]
X_test  = test.drop("outcome", axis=1).values
Y_test  = test["outcome"]

# Random Forest
random_forest = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, max_features=0.5, n_jobs=-1)
random_forest.fit(X_train, Y_train)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print('Acurácia do modelo RandomForestClassifier:',acc_random_forest,"\n")

Y_pred1 = random_forest.predict(X_test)

# Matrix de Confusão
print(pd.crosstab(Y_test,Y_pred1,
                  rownames=["Real"], 
                  colnames=["Predict"], 
                  margins=True))
                  
                  
                  # Decision Tree
decision_tree = DecisionTreeClassifier(max_depth = 3)
decision_tree.fit(X_train, Y_train)

Y_pred2 = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print('Acurácia do modelo DecisionTreeClassifier:',acc_decision_tree, "\n")

# Matrix de Confusão
print(pd.crosstab(Y_test,Y_pred2,
                  rownames=["Real"], 
                  colnames=["Predict"], 
                  margins=True))
                  
                  # KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)

Y_pred3 = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print('Acurácia do modelo KNeighborsClassifier:',acc_knn, "\n")

# Matrix de Confusão
print(pd.crosstab(Y_test,Y_pred3,
                  rownames=["Real"], 
                  colnames=["Predict"], 
                  margins=True))
                  
                  # Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print('Acurácia do modelo LogisticRegression:',acc_log, "\n")

Y_pred4 = logreg.predict(X_test)

# Matrix de Confusão
print(pd.crosstab(Y_test,Y_pred4,
                  rownames=["Real"], 
                  colnames=["Predict"], 
                  margins=True))
                  
                  # Ranking final do percentual de acurácia dos modelos aplicados
results = pd.DataFrame({
    'Model': ['Random Forest','Logistic Regression','KNN','Decision Tree'],
    'Score': [acc_random_forest, acc_log, acc_knn, acc_decision_tree]})
    
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)
