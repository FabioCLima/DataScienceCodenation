#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, KBinsDiscretizer,
    MinMaxScaler, StandardScaler
)
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)


# In[8]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[5]:


# Sua análise começa aqui.


# In[6]:


df = countries.copy()
df.info()


# Para substituir a vírgula por ponto, podemos aplicar funções Python para os elementos que precisam ser formatados do DataFrame. Isso pode ser executado através do applymap:

# In[7]:


format = lambda x: str(x).replace(',','.')


# In[9]:


df = countries.copy()
df = df.applymap(format)


# Para remover os espaços em branco usamos o str.strip()

# In[10]:


df['Country'] = df['Country'].str.strip()
df['Region'] = df['Region'].str.strip()


# In[11]:


lista = df.columns.tolist()
print(lista)


# In[13]:


df[['Population', 'Area']] = df[['Population', 'Area']].astype('int64')


# In[14]:


lista_floats = ['Pop_density', 'Coastline_ratio', 'Net_migration', 'Infant_mortality', 'GDP', 'Literacy', 'Phones_per_1000', 'Arable', 'Crops', 'Other',
                'Climate', 'Birthrate', 'Deathrate', 'Agriculture', 'Industry', 'Service']

df[lista_floats] = df[lista_floats].astype('float64')

df.dtypes


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# A função que será usada aqui é a unique, que devolve um array de valores únicos em um Series, encadeando com np.sort(obj).tolist().

# In[15]:


def q1():
    # Retorne aqui o resultado da questão 1.
    uniques = df['Region'].unique()
    uniques_sorted = np.sort(uniques).tolist()
    return uniques_sorted

q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[16]:


def q2():
    # Retorne aqui o resultado da questão 2.
    kbins = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy = 'quantile') # instância de discretização
    intervals = kbins.fit_transform(df[['Pop_density']]) # aplicação da discretização
    answer = int((intervals >= 9).sum())
    return answer
q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[17]:


pd.DataFrame({
    'unicos': df[['Region', 'Climate']].nunique(),
    'nan'   : df[['Region', 'Climate']].isnull().sum()
})


# In[18]:


def q3():
    # Retorne aqui o resultado da questão 3.
    one_hot = pd.get_dummies(df[['Region', 'Climate']].fillna('NaN'))
    answer = int(one_hot.shape[1])
    return answer

q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[19]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]

df_test_country = pd.DataFrame([test_country], columns = df.columns)
df_test_country.head()


# In[20]:


def q4():
    # Retorne aqui o resultado da questão 4.
    # Pipeline para atributos numéricos
    num_pipeline = Pipeline(
                           steps=[("imputer", SimpleImputer(strategy="median")),
                                  ("scaler" , StandardScaler())])
    df_num = df.select_dtypes(include = [np.number])   # seleção das colunas numéricas 
    cols_num = df_num.columns.tolist()
    df_num_tr = num_pipeline.fit(df_num)
    
    # test_country
    df_test_country_tr = num_pipeline.transform(df_test_country[cols_num])
    df_test_tr_pipeline = pd.DataFrame(df_test_country_tr, columns = df_num.columns)
  
    answer = float(df_test_tr_pipeline['Arable'].round(3))
    return answer

q4()   


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# Para responder essa questão temos que primeiramente analisar a distribuição dos dados, para tentar inferir o impacto de retirar esses outiliers.

# In[21]:


# Calculo dos quantis usados no método boxplot.
q25, q50, q75 = np.quantile(df['Net_migration'].dropna(), [0.25, 0.50, 0.75])

iqr = q75 - q25

df_q5 = df.copy()

df_q5['Outlier'] = 0
df_q5.loc[df_q5['Net_migration'] < (q25 - 1.5*iqr), 'Outlier'] = 1
df_q5.loc[df_q5['Net_migration'] > (q75 + 1.5*iqr), 'Outlier'] = 1


# In[22]:


fig, ax = plt.subplots()
ax.hist(df_q5['Net_migration'], bins = 40, color = 'orange', edgecolor = 'k')
ax.set_title(f'q25 = {q25.round(3)}, q75 = {q75.round(3)}, IQR = {iqr.round(3)}', fontsize = 16)
plt.show()
plt.clf()


# In[23]:


# Calcula da porcentagem de zeros
zeros = np.around(len(df[df['Net_migration'] == 0]) / (0.01*len(df)),2)
print(f'Porcentagem de 0s:{zeros}%')


# In[24]:


sns.boxplot(df['Net_migration'], orient = 'vertical')
plt.show()
plt.clf()


# In[25]:


def q5():
    # Retorne aqui o resultado da questão 5.
    net_migration = df['Net_migration'].dropna()
    
    limits_non_outlier = [q25 - 1.5*iqr, q75 + 1.5 * iqr]
    cutoff_lower, cutoff_upper = int((net_migration < limits_non_outlier[0]).sum()),int((net_migration > limits_non_outlier[1]).sum())
    
    cutoff = bool((cutoff_lower/len(net_migration)) < 0.05 or ((cutoff_lower/len(net_migration)) < 0.05))
    
    return cutoff_lower, cutoff_upper, cutoff

q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[26]:


from sklearn.datasets import fetch_20newsgroups

categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)


# In[27]:


def q6():
    # Retorne aqui o resultado da questão 6
    count_vectorizer = CountVectorizer()  # iniciando a instancia
    newsgroup_counts = count_vectorizer.fit_transform(newsgroup.data)
    words = pd.DataFrame(newsgroup_counts.toarray(), columns = count_vectorizer.get_feature_names())
    answer = int(words['phone'].sum())
    return answer

q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[28]:


def q7():
    # Retorne aqui o resultado da questão 7.
    tfidf_vec = TfidfVectorizer().fit(newsgroup.data)
    
    tfidf = tfidf_vec.transform(newsgroup.data)
    answer = float(tfidf[:,tfidf_vec.vocabulary_['phone']].sum().round(3))
    return answer
q7()

