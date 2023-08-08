#!/usr/bin/env python
# coding: utf-8

# #  BOSTON HOUSE PRICE PREDICTION
# 
# 

# In[ ]:


# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
df


# In[ ]:


df.isnull().sum()  #there are no null values


# In[ ]:


df.duplicated().sum() #no duplicate values


# In[ ]:


df.info()  #to gain insights on numerical and categorical columns presents in the dataset


# In[ ]:


df.describe()  #to get statistical insights


# In[ ]:


df.columns  #to find the feautures in the datasets


# #  dividing data into dependent and independent variables
# 

# In[ ]:


x=df.iloc[:,:-1]  #independent variables
x


# In[ ]:


y=df.iloc[:,-1]  #target variables or dependent variables
y


# In[ ]:


sns.pairplot(df)


# In[ ]:


plt.figure(figsize=(7,7))
sns.set(style="whitegrid")

plt.hist(y,bins=50)
plt.xlabel("MEDV",fontsize=18)
plt.ylabel("Frequency",fontsize=18)
plt.title("Price Distribution",fontsize=20)


# In[ ]:


sns.lmplot(x='rm',y='medv',data=df)


# In[ ]:


sns.distplot(y,bins=50)
plt.xlabel("MEDV",fontsize=18)
plt.ylabel("Frequency",fontsize=18)
plt.title("Price Distribution",fontsize=20)


# In[ ]:


#from the above graph we can see data is almost normally distributed


#now lets check co-relation

df.corr()['medv']


# In[ ]:


corr=df.corr()

plt.figure(figsize=(15,15))
sns.heatmap(corr,annot=True)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[ ]:


x_train.shape,x_test.shape


# In[ ]:


from sklearn.linear_model import LinearRegression

reg=LinearRegression()
reg.fit(x_train,y_train)


# In[ ]:


print("training R squared is",reg.score(x_train,y_train))
print("test R squared is",reg.score(x_test,y_test))


# In[ ]:


y_pred=reg.predict(x_test)


# # Model slope and intercept term
# 

# In[ ]:


a=reg.coef_
a

b=reg.intercept_
b


# # Calculate and print RMSE

# In[ ]:


from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
print("mse:",mse)
print("rmse value: {:.4f}".format(rmse))


# In[ ]:


from sklearn.metrics import r2_score
print("r2 scorevalue:{:.4f}".format(r2_score(y_test,y_pred)))


# # conclusion
# 
# 
# we can infer that model is a good fit as r2 score value is 0.732 ( r2 score value should be higher or close to 1)
# aslo training R squared and test R sqare values are approximately similar we can say that model is neither overfitting nor underfitting hence regularization techniques is not required
# 
# 

# In[ ]:




