#!/usr/bin/env python
# coding: utf-8

# Problem Definition
# The task here is to predict whether a bank currency note is authentic or not based on four attributes i.e. variance of the image wavelet transformed image, skewness, entropy, and curtosis of the image.
# 
# Solution
# This is a binary classification problem and we will use a random forest classifier to solve this problem. Steps followed to solve this problem will be similar to the steps performed for regression.

# In[11]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# In[12]:


dataset = pd.read_csv('C:\\Users\AmEy\personal\Downloads\expense_authentication.csv')
dataset.head()


# In[13]:


dataset.isna().sum()


# In[14]:


dataset.Class.unique()


# In[15]:


dataset.Class.value_counts()


# In[16]:


X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)


# In[18]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[19]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[20]:


accuracy = accuracy_score(y_test,y_pred)


# In[21]:


accuracy


# In case of regression we used the RandomForestRegressor class of the sklearn.ensemble library. For classification, we will RandomForestClassifier class of the sklearn.ensemble library. RandomForestClassifier class also takes n_estimators as a parameter. Like before, this parameter defines the number of trees in our random forest. We will start with 20 trees again. You can find details for all of the parameters of RandomForestClassifier here.

# ###### 6. Evaluating the Algorithm
# For classification problems the metrics used to evaluate an algorithm are accuracy, confusion matrix, precision recall, and F1 values. Execute the following script to find these values:
# In[22]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print('model accuracy:',accuracy_score(y_test, y_pred))


# The accuracy achieved for by our random forest classifier with 20 trees is 98.78%. Unlike before, changing the number of estimators for this problem didn't significantly improve the results, as shown in the following chart. Here the X-axis contains the number of estimators while the Y-axis shows the accuracy.
98.78% is a pretty good accuracy, so there isn't much point in increasing our number of estimators anyway. We can see that increasing the number of estimators did not further improve the accuracy.
# In[ ]:





# In[ ]:




