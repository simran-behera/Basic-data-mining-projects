#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pylab as pl
from sklearn import linear_model
from sklearn.metrics import r2_score


# In[2]:


a=pd.read_csv("C:\\Users\\PPS 30\\Downloads\\Data for Workshop\\FuelConsumption.csv")
d=pd.DataFrame(a)
d


# In[3]:


d.shape


# In[4]:


d.describe()


# In[5]:


df=d[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(df.shape)
print(df.head())
print(df.dtypes)


# In[21]:


#fuelconsumption_comb vs Co2emissions
plt.scatter(df.FUELCONSUMPTION_COMB,df.CO2EMISSIONS,color='RED')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2EMISSIONS")
plt.show()
#TO FIND CORRELATION COEFFICIENT
print(np.corrcoef(x=df.FUELCONSUMPTION_COMB,y=df.CO2EMISSIONS))
#ANOTHER WAY TO FIND CORREALTION COEFFICIENT
#print(np.corrcoef(x=df['FUELCONSUMPTION_COMB'],y=df['CO2EMISSIONS']))


# # correlation range [-1,1]
# 1 ->highly directly correlated
# -1->highly inversely correlated

# In[22]:


#Enginesize vs Co2emissions
plt.scatter(df.ENGINESIZE,df.CO2EMISSIONS,color='RED')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()
#TO FIND CORRELATION COEFFICIENT
print(np.corrcoef(x=df.ENGINESIZE,y=df.CO2EMISSIONS))


# In[23]:


#Cylinders vs Co2emissions
plt.scatter(df.CYLINDERS,df.CO2EMISSIONS,color='RED')
plt.xlabel("CYLINDERS")
plt.ylabel("CO2EMISSIONS")
plt.show()
#TO FIND CORRELATION COEFFICIENT
print(np.corrcoef(x=df.CYLINDERS,y=df.CO2EMISSIONS))


# In[24]:


msk=np.random.rand(len(df))<0.8
train=df[msk]
test=df[~msk]
print(train.head())
print(test.head())
print(train.shape)
print(test.shape)


# In[28]:


regr=linear_model.LinearRegression()
x_train=np.asanyarray(train[['ENGINESIZE']])
y_train=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x_train,y_train)


# In[27]:


print('coefficients:',regr.coef_)
print('intercept:',regr.intercept_)


# In[29]:


regr=linear_model.LinearRegression()
x_train=np.asanyarray(train[['CYLINDERS']])
y_train=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x_train,y_train)


# In[30]:


print('coefficients:',regr.coef_)
print('intercept:',regr.intercept_)


# In[31]:


regr=linear_model.LinearRegression()
x_train=np.asanyarray(train[['FUELCONSUMPTION_COMB']])
y_train=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x_train,y_train)


# In[32]:


print('coefficients:',regr.coef_)
print('intercept:',regr.intercept_)

