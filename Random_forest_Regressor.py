#!/usr/bin/env python
# coding: utf-8

# #### Problem Definition
# The problem here is to predict the fuel consumption (in millions of gallons) in 48 of the US states based on petrol tax (in cents), per capita income (dollars), paved highways (in miles) and the proportion of population with the driving license.

# ###### Part 1: Using Random Forest for Regression

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


dataset= pd.read_csv('.....\petrol_consumption.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.isnull().sum()


# In[5]:


dataset.describe().T


# ###### 3. Preparing Data For Training
Two tasks will be performed in this section. The first task is to divide data into 'attributes' and 'label' sets. The resultant data is then divided into training and test sets.

The following script divides data into attributes and labels:
# In[6]:


X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values


# In[7]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ###### 4. Feature Scaling

# Random forest is a tree based algotrithm,for which feature scaling is not mandatory

# ###### 5. Training the Algorithm

# Now  it is time to train our random forest algorithm to solve this regression problem. Execute the following code:

# In[8]:


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

The RandomForestRegressor class of the sklearn.ensemble library is used to solve regression problems via random forest. The most important parameter of the RandomForestRegressor class is the n_estimators parameter. This parameter defines the number of trees in the random forest. We will start with n_estimator=20 to see how our algorithm performs. You can find details for all of the parameters of RandomForestRegressor here.
# ###### 6. Evaluating the Algorithm
The last and final step of solving a machine learning problem is to evaluate the performance of the algorithm. For regression problems the metrics used to evaluate an algorithm are mean absolute error, mean squared error, and root mean squared error. Execute the following code to find these values:
# In[9]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# With 20 trees, the root mean squared error is 64.93 which is greater than 10 percent of the average petrol consumption i.e. 576.77. This may indicate, among other things, that we have not used enough estimators (trees).
# 
# If the number of estimators is changed to 200, the results are as follows:

# In[10]:


from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


# In[11]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# The following chart shows the decrease in the value of the root mean squared error (RMSE) with respect to number of estimators. Here the X-axis contains the number of estimators while the Y-axis contains the value for root mean squared error.
it can be see that the error values decreases with the increase in number of estimator. After 200 the rate of decrease in error diminishes, so therefore 200 is a good number for n_estimators. 