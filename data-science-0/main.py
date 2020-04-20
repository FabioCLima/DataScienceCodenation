#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[3]:


df = black_friday
df.head(2)


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[ ]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return df.shape    


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[ ]:


def q2():
    # Retorne aqui o resultado da questão 2.
    df2 = df[["Gender", "Age"]]
    df3 = df2[(df2["Gender"] == "F") & (df2["Age"] == "26-35")]
    return int(df3["Gender"].count())             


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[ ]:


def q3():
    # Retorne aqui o resultado da questão 3.
    df2 = df.drop_duplicates(subset = "User_ID")
    return int(df2["User_ID"].count())   


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[ ]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return df.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[ ]:


def q5():
    # Retorne aqui o resultado da questão 5.
    lt = df.shape[0]
    df_notna = df.dropna(axis = 0)
    lnotna = df_notna.shape[0]
    return float((lt - lnotna)/lt)


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[ ]:


def q6():
    # Retorne aqui o resultado da questão 6.
    return df.isnull().sum().max()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[ ]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return int(df["Product_Category_3"].dropna().mode())


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[ ]:


def q8():
    # Retorne aqui o resultado da questão 8.
    def norm(column):
        return ((column - column.min())/(column.max() - column.min()))
        
    return (df["Purchase"].agg(norm).mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variável `Purchase` após sua padronização? Responda como um único escalar.

# In[ ]:


def q9():
    # Retorne aqui o resultado da questão 9.
    def standard(column):
        return ((column - column.mean())/
                column.std())
    
    df["Purchase_std"] = df["Purchase"].agg(standard)
    
    answer9 = df.loc[(df["Purchase_std"] >= -1) & 
                     (df["Purchase_std"] <=  1),:].shape[0]
    return answer9


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[ ]:


def q10():
    # Retorne aqui o resultado da questão 10.
    df_booleano = df[["Product_Category_2", "Product_Category_3"]].isna()
    
    dfp2 = df_booleano.loc[df_booleano["Product_Category_2"] == True, :]
    
    answer10 = dfp2["Product_Category_2"].equals(dfp2["Product_Category_3"])
    return answer10

