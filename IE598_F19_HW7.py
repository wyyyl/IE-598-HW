#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import timeit
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


# In[29]:


data =pd.read_csv("ccdefault.csv",header = 0)
X = data.iloc[:,1:24]
y = data.iloc[:,-1]
print(X.shape,y.shape)


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=1)
insample=[]
time=[]
score=[]
for i in [1,10,30,50,100]:
    start=timeit.default_timer()
    rfc = RandomForestClassifier(n_estimators=i, n_jobs=2)
    rfc.fit(X_train, y_train)
    train_acc =np.mean(cross_val_score(rfc,X_train,y_train,cv=10))
    insample.append(train_acc)
    end=timeit.default_timer()
    time.append(end-start)

for i in range(0,5):
    score.append([insample[i],time[i]])
df=pd.DataFrame(score, columns=["in sample accuracy","running time"]) 
df.rename({0:"1",1:"10",2:"30",3:"50",4:"100"},axis=0)


# In[36]:


rfc_tuning = RandomForestClassifier(n_estimators=100,random_state=1)
rfc_tuning.fit(X_train, y_train)
feat_labels = X.columns.get_values()
importances = rfc_tuning.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) Feature # %d %s (%f)" % (f + 1, indices[f],feat_labels[indices[f]],importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),importances[indices],align='center')
plt.xticks(range(X_train.shape[1]),feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()


# In[41]:


print("My name is {yulong wang}")
print("My NetID is: {yulongw2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:




