# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


# %%
#reading the csv file
df=pd.read_csv('Iris.csv')
df

# %%
df.info()
df.shape

# %%
df.isnull().sum()

# %%
df.duplicated().sum()

# %%
#check datset is balanced or not
df["Species"].value_counts()

# %%
#data visualization

sns.pairplot(df, hue='Species')

# %%
# seperate features and target class
data=df.values
x=data[:,1:5]
x
y=data[:,5]
y

# %%
#split the data into training and test dataset

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)

# %%
# Model1:Svm

from sklearn.svm import SVC
clf=SVC()
clf.fit(x_train,y_train)

# %%
y_test_pred=clf.predict(x_test)
y_train_pred = clf.predict(x_train)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_test_pred))

# %%
from sklearn import metrics
metrics.confusion_matrix(y_test,y_test_pred)

# %%
metrics.classification_report(y_test,y_test_pred)

# %%
#method 2 using logistic regression

from sklearn.linear_model import LogisticRegression
clf1=LogisticRegression()
clf1.fit(x_train,y_train)

# %%
y_test_pred1 =clf1.predict(x_test)
metrics.accuracy_score(y_test,y_test_pred1)

# %%
from sklearn import metrics
metrics.confusion_matrix(y_test,y_test_pred1)

# %%
from sklearn import metrics
metrics.classification_report(y_test,y_test_pred1)

# %%
#method 3 using decession Tree
from sklearn.tree import DecisionTreeClassifier
dtc= DecisionTreeClassifier()
dtc.fit(x_train,y_train)

# %%
y_test_pred2=dtc.predict(x_test)
metrics.accuracy_score(y_test,y_test_pred2)

# %%
metrics.confusion_matrix(y_test,y_test_pred2)

# %%
metrics.classification_report(y_test,y_test_pred2)


