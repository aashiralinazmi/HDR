#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.metrics import classification_report
import joblib
from sklearn import datasets


# In[2]:


#getting MNIST of size 70k images
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
X = np.array(mnist.data)
Y = np.array(mnist.target)
X =  X.astype('float32')


# In[3]:


#getting Our Test Data
X_test,Y_test = X[60000:], Y[60000:]


# In[4]:


#Normalizing Our Features
X_test = X_test /255


# In[5]:


#loading our saved model
model = joblib.load('model.pkl')


# In[6]:


#predicting
y_pred = model.predict(X_test)


# In[7]:



print(classification_report(y_pred,Y_test))


# In[ ]:





# In[ ]:




