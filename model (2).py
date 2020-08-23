#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# In[2]:


dataset = pd.read_csv('salary.csv')


# In[3]:


dataset['experience'].fillna(0, inplace=True)


# In[4]:


X = dataset.iloc[:, :3]


# In[5]:


def convert_to_int(word):
    word_dict = {'One':1, 'Two':2, 'Three':3, 'Four':4, 'Five':5, 'Six':6, 'Seven':7, 'Eight':8,
                 'zero':0, 0: 0}
    return word_dict[word]


# In[6]:


X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))


# In[7]:


y = dataset.iloc[:, -1]


# In[8]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


# In[9]:


regressor.fit(X, y)


# In[10]:


pickle.dump(regressor, open('model.pkl','wb'))


# In[11]:


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))


# In[ ]:




