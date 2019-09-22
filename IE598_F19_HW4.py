#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import data and drop all unusable rows
import pandas as pd
h=pd.read_excel('housing.xlsx')
H=h.dropna()
print(H.columns)
H.head()


# In[2]:


H.info()


# In[15]:


#Scatterplot of all variables
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(H,height=2.5)
plt.tight_layout()
plt.show()


# In[4]:


#Scatterplot of all variables that are highly correlated with MEDV
cols=H[['LSTAT','PTRATIO','RM','MEDV']]
sns.pairplot(cols)
plt.show()


# In[18]:


#Correlation Coefficient Matrix
import numpy as np
corr_mat=pd.DataFrame(H.corr())
corr_mat


# In[6]:


#heatmap
cm=np.corrcoef(H.values.T)
sns.set(font_scale=1.5)
hm=sns.heatmap(cm,cbar=True,annot=False,square=True,fmt='.2f',annot_kws={'size':15},yticklabels=H.columns,xticklabels=H.columns)
plt.show()


# In[7]:


#Split data 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X=H.iloc[:,:-1].values
y=H['MEDV'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(H.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[8]:


#LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
reg=LinearRegression()
reg.fit(X_train,y_train)
print("Coefficients:",reg.coef_)
print('Intercept: %.3f' % reg.intercept_)
y_pred=reg.predict(X_test)
print("R^2: {}".format(reg.score(X_test,y_test)))
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error: {}".format(mse))


# In[9]:


y_train_pred=reg.predict(X_train)
y_test_pred=reg.predict(X_test)
_=plt.scatter(y_train_pred,y_train_pred-y_train,c='steelblue',marker='o',edgecolor='white',label='Training data')
_=plt.scatter(y_test_pred,y_test_pred-y_test,c='limegreen',marker='s',edgecolor='white',label='Test data')
_=plt.xlabel('Predicted values')
_=plt.ylabel('Residuals')
_=plt.legend(loc='upper left')
_=plt.hlines(y=0,xmin=-10,xmax=50,color='black',lw=2)
_=plt.xlim([-10,50])
plt.show()


# In[10]:


#LASSO
from sklearn.linear_model import Lasso
H1=pd.read_csv('housing2.csv')
H1=H1.dropna()
X=H1.iloc[:,:-1].values
y=H1['MEDV'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
alpha_space=np.logspace(-5,-0.25,50)
alpha_value=[]
R2_value=[]
for alpha in alpha_space:
    lasso=Lasso(alpha=alpha,normalize=True)
    lasso.fit(X_train,y_train)
    R2=lasso.score(X_test,y_test)
    alpha_value.append(alpha)
    R2_value.append(R2)
best=alpha_space[R2_value.index(max(R2_value))]
print("Best Alpha:{}".format(best))
plt.plot(alpha_value,R2_value)
plt.xlabel('alpha')
plt.ylabel('R2')
plt.xlim(0,0.5)
plt.xticks(np.arange(0, 0.55, step=0.05))
plt.show()


# In[11]:


lasso=Lasso(alpha=0.0008685,normalize=True)
lasso.fit(X_train,y_train)
print("Coefficients:",lasso.coef_)
print('Intercept: %.3f' % lasso.intercept_)
y_pred=lasso.predict(X_test)
print("R^2: {}".format(lasso.score(X_test,y_test)))
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error: {}".format(mse))
y_train_pred=lasso.predict(X_train)
y_test_pred=lasso.predict(X_test)
plt.scatter(y_train_pred,y_train_pred-y_train,c='steelblue',marker='o',edgecolor='white',label='Training data')
plt.scatter(y_test_pred,y_test_pred-y_test,c='limegreen',marker='s',edgecolor='white',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,color='black',lw=2)
plt.xlim([-10,50])
plt.show()


# In[12]:


#RIDGE
from sklearn.linear_model import Ridge
alpha_space=np.logspace(-5,-0.25,50)
mse_value=[]
alpha_value=[]
R2_value=[]
for alpha in alpha_space:
    ridge=Ridge(alpha=alpha,normalize=True)
    ridge.fit(X_train,y_train)
    R2=ridge.score(X_test,y_test)
    alpha_value.append(alpha)
    R2_value.append(R2)
best=alpha_space[R2_value.index(max(R2_value))]
print("Best Alpha:{}".format(best))
plt.plot(alpha_value,R2_value)
plt.xlabel('alpha')
plt.ylabel('R2')
plt.xlim(0,0.5)
plt.xticks(np.arange(0, 0.55, step=0.05))
plt.show()


# In[13]:


ridge=Ridge(alpha=0.060341,normalize=True)
ridge.fit(X_train,y_train)
print("Coefficients:",ridge.coef_)
print('Intercept: %.3f' % ridge.intercept_)
y_pred=ridge.predict(X_test)
print("R^2: {}".format(ridge.score(X_test,y_test)))
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error: {}".format(mse))
y_train_pred=ridge.predict(X_train)
y_test_pred=ridge.predict(X_test)
plt.scatter(y_train_pred,y_train_pred-y_train,c='steelblue',marker='o',edgecolor='white',label='Training data')
plt.scatter(y_test_pred,y_test_pred-y_test,c='limegreen',marker='s',edgecolor='white',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,color='black',lw=2)
plt.xlim([-10,50])
plt.show()


# In[14]:


print("My name is Yulong Wang")
print("My NetID is yulongw2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation")

