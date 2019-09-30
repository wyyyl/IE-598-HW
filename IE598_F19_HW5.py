#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn.svm import SVR


# In[3]:


#EDA
da=pd.read_csv('hw5_treasury yield curve data.csv')
data=da.dropna()
data=data.iloc[:,1:32]
print('Number of observations:',data.shape[0])
print('Number of variables:',data.shape[1])
data.describe()


# In[4]:


corr_mat=pd.DataFrame(data.corr())
corr_mat


# In[5]:


data1=data[['SVENF12','SVENF13','SVENF14','SVENF15','SVENF16','SVENF17','Adj_Close']]
cm=np.corrcoef(data1.values.T)
sns.set(font_scale=1)
hm=sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},yticklabels=data1.columns,xticklabels=data1.columns)
plt.show()
sns.pairplot(data1,height=2)
plt.show()


# In[6]:


stats.probplot(data.Adj_Close,dist="norm",plot=pylab)
pylab.show()


# In[7]:


#PCA
X=data.iloc[:,0:30]
y=data.iloc[:,30]
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.15,random_state=42)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
model=PCA(n_components=None)
X_train_pca=model.fit_transform(X_train_std)
X_test_pca=model.transform(X_test_std)
cov_mat = np.cov(X_train_pca.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1,31), var_exp, alpha=0.5, align='center',label='individual explained variance')
plt.step(range(1,31), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()


# In[14]:


model_new=PCA(n_components=3)
X_train_pca3=model_new.fit_transform(X_train_std)
X_test_pca3=model_new.fit_transform(X_test_std)
print("The explained variance ratio of the 3 component pca model is:", model_new.explained_variance_ratio_)
cum_var_exp_ratio=np.cumsum(model_new.explained_variance_ratio_)
print("The cumulative explained variance ratio of the 3 component pca model is:", cum_var_exp_ratio)
print("The explained variance of the 3 component pca model is:", model_new.explained_variance_)
cum_var_exp=np.cumsum(model_new.explained_variance_)
print("The cumulative explained variance of the 3 component pca model is:", cum_var_exp)
plt.bar(range(1,4), model_new.explained_variance_ratio_, alpha=0.5, align='center',label='individual explained variance')
plt.step(range(1,4), cum_var_exp_ratio, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()


# In[8]:


#linear classifier model
lr = LogisticRegression()
lr=lr.fit(X_train,y_train)


# In[15]:


get_ipython().run_cell_magic('time', '', '#linear regression model\nln = LinearRegression()\nln.fit(X_train,y_train)\nR2_train=ln.score(X_train,y_train)\nR2_test=ln.score(X_test,y_test)\nprint("R2 score for linear regression model in original training sample is",R2_train)\nprint("R2 score for linear regression model in original testing sample is",R2_test)\ny_train_pred=ln.predict(X_train)\ny_test_pred=ln.predict(X_test)\nRMSE_train=np.sqrt(mean_squared_error(y_train,y_train_pred))\nRMSE_test=np.sqrt(mean_squared_error(y_test,y_test_pred))\nprint("RMSE score for linear regression model in original training sample is",RMSE_train)\nprint("RMSE score for linear regression model in original testing sample is",RMSE_test)\nln.fit(X_train_pca3,y_train)\nR2_train_pca3=ln.score(X_train_pca3,y_train)\nR2_test_pca3=ln.score(X_test_pca3,y_test)\nprint("R2 score for linear regression model in PCA transformed training sample is",R2_train_pca3)\nprint("R2 score for linear regression model in PCA transformed testing sample is",R2_test_pca3)\ny_train_pred_pca3=ln.predict(X_train_pca3)\ny_test_pred_pca3=ln.predict(X_test_pca3)\nRMSE_train_pca3=np.sqrt(mean_squared_error(y_train,y_train_pred_pca3))\nRMSE_test_pca3=np.sqrt(mean_squared_error(y_test,y_test_pred_pca3))\nprint("RMSE score for linear regression model in PCA transformed training sample is",RMSE_train_pca3)\nprint("RMSE score for linear regression model in PCA transformed testing sample is",RMSE_test_pca3)')


# In[16]:


get_ipython().run_cell_magic('time', '', '#SVR model\nsvm = SVR(gamma=\'auto\')\nsvm.fit(X_train,y_train)\nR2_train_svm=svm.score(X_train,y_train)\nR2_test_svm=svm.score(X_test,y_test)\nprint("R2 score for support vector regression model in original training sample is",R2_train)\nprint("R2 score for support vector regression model in original testing sample is",R2_test)\ny_train_pred_svm=svm.predict(X_train)\ny_test_pred_svm=svm.predict(X_test)\nRMSE_train_svm=np.sqrt(mean_squared_error(y_train,y_train_pred_svm))\nRMSE_test_svm=np.sqrt(mean_squared_error(y_test,y_test_pred_svm))\nprint("RMSE score for support vector regression model in original training sample is",RMSE_train_svm)\nprint("RMSE score for support vector regression model in original testing sample is",RMSE_test_svm)\nsvm.fit(X_train_pca3,y_train)\nR2_train_pca3_svm=svm.score(X_train_pca3,y_train)\nR2_test_pca3_svm=svm.score(X_test_pca3,y_test)\nprint("R2 score for support vector regression model in PCA transformed training sample is",R2_train_pca3_svm)\nprint("R2 score for support vector regression model in PCA transformed testing sample is",R2_test_pca3_svm)\ny_train_pred_pca3_svm=svm.predict(X_train_pca3)\ny_test_pred_pca3_svm=svm.predict(X_test_pca3)\nRMSE_train_pca3_svm=np.sqrt(mean_squared_error(y_train,y_train_pred_pca3_svm))\nRMSE_test_pca3_svm=np.sqrt(mean_squared_error(y_test,y_test_pred_pca3_svm))\nprint("RMSE score for support vector regression model in PCA transformed training sample is",RMSE_train_pca3_svm)\nprint("RMSE score for support vector regression model in PCA transformed testing sample is",RMSE_test_pca3_svm)')


# In[17]:





# In[ ]:




