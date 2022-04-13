#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing required modules

import joblib
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_digits
#for creating Neural Network  I am using  MLPClassifier from sklearn

from sklearn.neural_network import MLPClassifier


# In[2]:


#getting MNIST of size 70k images
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
X = np.array(mnist.data)
Y = np.array(mnist.target)
X =  X.astype('float32') 


# In[3]:


#splitting Dataset into Training and Testing dataset
#First 60k instances are for Training and last 10k are for testing
X_train, X_test = X[:60000], X[60000:]
Y_train, Y_test = Y[:60000], Y[60000:]


# In[4]:


X_train = X_train /255
X_test = X_test /255


# In[5]:


#creating Neural Network
# Neural Network has one hidden layer with 240 units
# Neural NetWork is of size 784-240-10

mlp = MLPClassifier(hidden_layer_sizes=(240), max_iter=500, verbose=True)


# In[6]:


#fitting our model
mlp.fit(X_train, Y_train)


# In[7]:


print("Training set score: %f" % mlp.score(X_train, Y_train))
print("Test set score: %f" % mlp.score(X_test, Y_test))     


# In[8]:


#saving our model
joblib.dump(mlp, "model.pkl")


# In[ ]:




