#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[2]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[3]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[4]:


# Sua análise da parte 1 começa aqui.
df = dataframe.copy()
df.head()


# In[5]:


df.describe().T


# In[6]:


# Como são graficamente as duas distribuições
sns.set(style = 'darkgrid', palette = 'bright')
i = 1
plt.figure(figsize = (20,20))
for c in df.describe().columns:
    plt.subplot(4, 3, i)
    plt.title(f"Histogram of {c}", fontsize = 16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.hist(df[c], bins = 20, color = 'green', edgecolor = 'k')
    i += 1
plt.show()    


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[7]:


def q1():
    # Retorne aqui o resultado da questão 1.
    """Calcula a diferença entre os quantis da normal e os quantis da binomial"""
    # Cálculo da diferença entre os quantis
    diff = (df.normal.quantile([0.25, 0.5, 0.75]) - df.binomial.quantile([0.25, 0.5, 0.75]))
    answer_q1 = tuple(round(diff, 3))
    return answer_q1


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
#   Devido ao número de amostras n = 10³, tinha inferido que a diferença ia ser pequena.
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?
#   Apesar de serem distribuições discreta - binomial, e normal - contínua, temos duas situações ocorrendo em paralelo, a lei dos grandes números e o teorema do limite central que a soma de variáveis aleatórias tendem a uma distribuição quando o número de amostras tende ao infinito.

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[8]:


def q2():
    # Retorne aqui o resultado da questão 2.
    """Cálculo da probabilidade pela ECDF sobre um intervalo dado"""
    ecdf = ECDF(df.normal)
    mu, sigma = df.normal.mean(), df.normal.std()
    prob_interval = float(ecdf(mu + sigma) - ecdf(mu - sigma))
    answer_q2 = round(prob_interval, round(3))
    return answer_q2


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
#   Sim, já que está dentro da regra "69-95-99.7".
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[9]:


def q3():
    # Retorne aqui o resultado da questão 3.
    """Calcula a diferença entre as médias e variâncias para das distribuições normal e binomial respectivamente"""
    mu_norm, mu_binom = df.mean()
    var_norm, var_binom = df.var()
    
    answer_q3 = np.round(mu_binom - mu_norm, 3), np.round(var_binom - var_norm, 3)
    return answer_q3
    


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
#   Sim, por conta do teorema do limite central.
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?
#   lei grandes números, no limite a binomial se tornará muito próxima da normal.

# ## Parte 2

# ### _Setup_ da parte 2

# In[10]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[11]:


# Sua análise da parte 2 começa aqui.
stars.head()


# In[13]:


stars.shape


# In[15]:


stars.describe().T


# In[16]:


sns.set(style = "ticks", color_codes = True)
g = sns.pairplot(stars, vars = ['mean_profile', 'sd_profile', 'kurt_profile', 'skew_profile', 'mean_curve', 
                                'sd_curve', 'kurt_curve', 'skew_curve'], plot_kws=dict(s=30, edgecolor = 'b', 
                                                                                       linewidth = 1),
                hue = 'target', markers = ["o", "s"], diag_kind = 'kde', diag_kws = dict(shade=True))


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[ ]:


def q4():
    # Retorne aqui o resultado da questão 4.
    # filtragem, padronização e calculo de ECDF
    # 1 - filtragem
    filtro = stars[stars['target'] == 0]['mean_profile']
    # 2 - standarization
    stan = (filtro - filtro.mean()) / filtro.std()
    # 3 - ECDF
    ecdf = ECDF(stan)
    # Quantiles
    quantis = sct.norm.ppf([0.8, 0.90, 0.95], loc = 0, scale = 1)
    q80, q90, q95 = quantis[0], quantis[1], quantis[2]
    answer_q4 = tuple(ecdf([quantis[0], quantis[1], quantis[2]]).round(3))
    answer_q4
    return answer_q4


# Para refletir:
# 
# * Os valores encontrados fazem sentido? Sim.
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
#   Que a padronização fez com que a distribuição que era assimétrica tornar-se simétrica. Após este procedimento, os quantis tendem a ficar próximos a probabilidade.

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[1]:


def q5():
    # Retorne aqui o resultado da questão 5.
    # filtragem 
    filter = stars[stars['target'] == 0]['mean_profile']
    # standarization
    standarization = sct.zscore(filter)
    # quantiles
    quantis = np.percentile(standarization, [25, 50, 75])
    # quantis da distribuição normal
    quantis_norm = sct.norm.ppf([0.25, 0.50, 0.75])
    answer_q5 = tuple(np.round(quantis - quantis_norm, 3))
    return answer_q5


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
#  sim.
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
#  Que ela tem uma distribuição próxima da normal.
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
