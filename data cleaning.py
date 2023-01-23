#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np

a=pd.read_csv("C:\\Users\\PPS 30\\Downloads\\Data for Workshop\\automobile.csv")
d=pd.DataFrame(a)
d


# In[12]:


d.head(10)


# In[10]:


d.shape


# In[11]:


d.columns


# In[14]:


d.dtypes


# In[20]:


d.replace('?',np.NaN,inplace=True)
d.head(10)


# In[22]:


d.dtypes


# In[38]:


mv=d.isnull()
print(mv.head(70))


# In[25]:


nmv=d.notnull()
print(nmv.head(10))


# In[30]:


for column in mv.columns.values.tolist():
    print(column)
    print(mv[column].value_counts())
    print("")


# In[32]:


#replacing with mean
avg=d['normalized-losbses'].astype('float').mean(axis=0)
print(avg)
d['normalized-losses'].replace(np.NaN,avg,inplace=True)
d


# In[39]:


#replacing with mean
avg=d['bore'].astype('float').mean(axis=0)
print(avg)
d['bore'].replace(np.NaN,avg,inplace=True)
d.head(60)


# In[40]:


#replacing with mean
avg=d['stroke'].astype('float').mean(axis=0)
print(avg)
d['stroke'].replace(np.NaN,avg,inplace=True)
d.head(60)


# In[44]:


#replacing with mean
avg=d['horsepower'].astype('float').mean(axis=0)
print(avg)
d['horsepower'].replace(np.NaN,avg,inplace=True)
d.head(100)


# In[53]:


#replacing with mean
avg=d['peak-rpm'].astype('float').mean(axis=0)
print(avg)
d['peak-rpm'].replace(np.NaN,avg,inplace=True)
d.head(100)


# In[45]:


#replacing with frequency
d['num-of-doors'].value_counts()


# In[47]:


d['num-of-doors'].value_counts().idxmax()


# In[46]:


d['num-of-doors'].replace(np.NaN,'four',inplace=True)
d.head(60)


# In[48]:


#dropping null values
d.dropna(subset=["price"],axis=0,inplace=True)
d


# In[51]:


d.reset_index(drop=True,inplace=True)


# In[57]:


d.head(20)


# In[55]:


mv=d.isnull()
print(mv.head(70))


# In[56]:


for column in mv.columns.values.tolist():
    print(column)
    print(mv[column].value_counts())
    print("")
    


# In[ ]:





# In[ ]:




