#!/usr/bin/env python
# coding: utf-8

# # <font color=red>Presented By Aditya Sikhwal</font>

# ### *Data Science and Business Analytics*
# ### *Task 01 - Prediction using Supervised ML*

# 
# 
# ### **Importing necessary libraries**
# 
# 

# In[47]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# 
# 
# ### **Exploring the data**

# In[48]:


data = pd.read_csv('https://bit.ly/w-data')
data.head()


# In[49]:


data.shape


# In[50]:


data.info()


# In[51]:


data.dtypes


# In[52]:


data.describe()


# 
# 
# ### **Visualizing and inspecting the data in a 2D plot**

# In[53]:


plt.figure(figsize=(10,5))
plt.title('Scores vs Hours',size=20)  
plt.xlabel('STUDY HOURS',size=15)  
plt.ylabel('SCORE',size=15) 
plt.scatter(data.Hours,data.Scores,color='GREEN')
plt.show()


# In[54]:


data.corr()


# 
# 
# ### **Preparing the data for training**

# In[55]:


x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# 
# 
# ### **Training Model**

# In[56]:


regressor = LinearRegression()
regressor.fit(x_train, y_train)

print('Model Training Complete...')


# ### **Plotting the regression line**

# In[57]:


line=regressor.coef_*x+regressor.intercept_
plt.scatter(x,y,color='green',marker='o')
plt.plot(x,line,color='red');
plt.title('Graphical relationship between the no. of Study hours and Scores obtained')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Scored(%)')
plt.show()


# In[58]:


print('intercept={}, slope coefficient={}'.format(regressor.intercept_,regressor.coef_))


# ### **Evaluating the model performance**

# In[59]:


y_pred = regressor.predict(x_test)

df = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
df


# ### **Accuracy Score**

# In[60]:


print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# ### **Building a Predictive System**

# In[61]:


hours = 9.25
pred = regressor.predict(np.array([hours]).reshape(-1, 1))
print(f"Hours Studied: {hours}")
print(f"Predected Score: {pred[0]:.0f}")

