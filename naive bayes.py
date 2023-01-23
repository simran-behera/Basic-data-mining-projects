#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score


# In[12]:


df= pd.read_csv("C:\\Users\\AIML-DAA22\\Downloads\\train.csv")
print(df.head(2))


# In[13]:


df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
df= df.replace({"Gender":{"Male":1, "Female":0 }})
df =df.replace({"Married" :{"Yes":1, "No":0}})
df['Dependents'] = df['Dependents'].replace('3+', '3')
df= df.replace({"Self_Employed":{"Yes":1, "No":0 }})
df= df.replace({"Education":{"Graduate":1, "Not Graduate":0 }})
df = df.drop(columns=['Loan_ID'])
df['Property_Area'] = df['Property_Area'].map({'Rural': 0, 'Urban': 1, 'Semiurban':2})


# In[14]:


df.fillna(df.median(), inplace=True)


# In[15]:


sns.heatmap(df.corr(),annot=True)
plt.show()


# In[18]:


df=df[['Married','Education','CoapplicantIncome','Credit_History','Property_Area','Loan_Status']]
train=df[['Married','Education','CoapplicantIncome','Credit_History','Property_Area']]
test=df[['Loan_Status']]


# In[19]:


X_train, X_val, Y_train, Y_val = train_test_split(train, test, test_size=0.2, random_state=1)


# In[26]:


X_train1, X_train2, Y_train1, Y_train2 = train_test_split(X_train, Y_train, test_size=0.3, random_state=12)


# In[28]:


classifier = GaussianNB()
classifier.fit(X_train2,Y_train2)
classifier.partial_fit(X_train1,Y_train1)


# In[36]:


def cross_validate(estimator, train, validation):
    X_train = train[0]
    Y_train = train[1]
    X_val = validation[0]
    Y_val = validation[1]
    train_predictions = classifier.predict(X_train)
    train_accuracy = accuracy_score(train_predictions, Y_train)
    train_recall = recall_score(train_predictions, Y_train)
    train_precision = precision_score(train_predictions, Y_train)
    val_predictions = classifier.predict(X_val)
    val_accuracy = accuracy_score(val_predictions, Y_val)
    val_recall = recall_score(val_predictions, Y_val)
    val_precision = precision_score(val_predictions, Y_val)
    print('Classification report')
    print('Accuracy Train: %.2f, Validation: %.2f' % (train_accuracy, val_accuracy))
    print('Recall Train: %.2f, Validation: %.2f' % (train_recall, val_recall))
    print('Precision Train: %.2f, Validation: %.2f' % (train_precision, val_precision))


# In[37]:


cross_validate(classifier, (X_train, Y_train), (X_val, Y_val))

