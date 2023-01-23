#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pylab as pl
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import log_loss
import itertools
import warnings
warnings.filterwarnings("ignore")


# In[18]:


a=pd.read_csv("C:\\Users\\PPS 30\\Downloads\\Data for Workshop\\ChurnData.csv")
d=pd.DataFrame(a)
d


# In[19]:


d.shape


# In[13]:


d.dtypes


# In[20]:


mv=d.isnull()
print(mv)


# In[21]:


d=d[['tenure','age','address','income','ed','employ','equip','churn']]
print(d.head())


# In[22]:


d['churn']=d['churn'].astype('int')
print(d.head())
print(d.shape)


# In[24]:


X=np.asarray(d[['tenure','age','address','income','ed','employ','equip','churn']])
print(X[0:5])
y=np.asarray(d['churn'])
print(y[0:5])


# In[26]:


X=preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])


# In[27]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
print("trainset:",X_train.shape,y_train.shape)
print("testset:",X_test.shape,y_test.shape)


# In[28]:


LR=LogisticRegression(C=0.01,solver='lbfgs')
LR.fit(X_train,y_train)
LR


# In[29]:


yhat=LR.predict(X_test)
print(yhat[0:5])


# In[30]:


yhat_prob=LR.predict_proba(X_test)
print(yhat_prob)


# In[31]:


confusion_matrix(y_test,yhat,labels=[1,0])

