#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[2]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA

from loguru import logger


# In[3]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[4]:


fifa = pd.read_csv("fifa.csv")


# In[5]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[8]:


# Sua análise começa aqui.
df = fifa.copy()


# In[9]:


df.head()


# In[10]:


df.shape


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[11]:


df.info()


# In[12]:


pd.DataFrame({
    'unique':df.nunique(),
    'nan'   :df.isnull().sum()
})


# In[13]:


## Como a quantidade de dados faltantes é pequena - podemos 
## podemos descartar sem maiores impactos no restante da 
## análise

df.dropna(axis = 0, inplace = True)
df.shape


# In[24]:


df.columns


# In[14]:


plt.figure(figsize = (20, 20))
matrix = df.corr()
sns.heatmap(matrix[(matrix > 0.7) | (matrix < -0.7)].round(2), annot = True)


# In[25]:


df.describe().T


# In[15]:


def q1():
    pc1 = PCA(n_components = 1).fit(df)
    pc1_var = pc1.explained_variance_ratio_
    return pc1_var.round(3).item()

q1()


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[16]:


def q2():
    pca = PCA().fit(df)
    explained_var_ratio = pca.explained_variance_ratio_
    plt.plot(np.cumsum(explained_var_ratio))
    plt.plot(np.linspace(0.95,0.95,40), ls="--", color="red")
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')

    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    n_components95 = len(cumulative_variance_ratio[cumulative_variance_ratio<=0.95]) + 1
    
    return n_components95

q2()  


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[18]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# Dado que as dimensões do dataframe me são:
# df.shape = (18159, 37)
# número de features = 37
# número de amostras = 18159
# 
# O problema pede para reduzirmos para duas dimensões d = 2:
# pca = PCA(n_components = 2)
# pca.fit(df)

# In[19]:


def q3():
    # reduzindo para d = 2 (dimensões)
    pca = PCA(n_components = 2)
    pca.fit(df)
    
    # pegar as duas componentes principais
    matrix_phi = pca.components_ 
    x_pc_components = np.matmul(matrix_phi,np.asarray(x))
    print(x_pc_components)
    return tuple(x_pc_components.round(3))
   
q3()


# In[20]:


''''backward stepwise selection = RFE

the goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features.

First, the estimator is trained on the initial set of features and the importance of each feature is obtained. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.
    estimator : A supervised learning estimator with a fit method that provides information about feature importance either through a coef_ attribute or   through a feature_importances_ attribute.

    n_features_to_select : The number of features to select. If None, half of the features are selected.

    step : If greater than or equal to 1, then step corresponds to the (integer) number of features to remove at each iteration. If within (0.0, 1.0), then step corresponds to the percentage (rounded down) of features to remove at each iteration.


'''
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error as mse


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[22]:


def q4():
    
    X = df.copy().drop(columns = ["Overall"])
    y = df["Overall"]

    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, shuffle = True)

    reg = LinearRegression()
    reg.fit(X_train, y_train)

    print("Model r2 score Linear regression:", reg.score(X_test, y_test))
    y_pred = reg.predict(X_test)
    print( 'MSE',  mse(y_test, y_pred))
    print('RMSE',  mse(y_test, y_pred, squared=False))
    print(pd.DataFrame.from_dict(dict(zip(X_train.columns, reg.coef_)), orient='index', columns=['coef']).sort_values(by='coef', ascending=False).head(5))

    selector = RFE(estimator = reg, n_features_to_select=5, step=1, verbose = 0)
    selector = selector.fit(X_train, y_train)
    selected_features5 = list(X_train.columns[selector.get_support()])
    print('\nMost important features RFE',selected_features5)
    print("\nModel r2 score RFE Linear regression selected features:", selector.score(X_test, y_test))
    y_pred = selector.predict(X_test)
    print( 'MSE',  mse(y_test, y_pred))
    print('RMSE',  mse(y_test, y_pred, squared=False))

    X_train5 = selector.transform(X_train)
    reg.fit(X_train5, y_train)
    coeficients = reg.coef_ 
    print(pd.DataFrame.from_dict(dict(zip(selected_features5, reg.coef_)), orient='index', columns=['coef']).sort_values(by='coef', ascending=False).head(5))
    
    # plt.scatter(y_test,y_pred)
    # plt.show()
    return selected_features5

q4()

