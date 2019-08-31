import sklearn
print( 'The scikit learn version is {}.'.format(sklearn.__version__))
#check scikit-learn version
#The scikit learn version is 0.21.2.

from sklearn import datasets
iris = datasets.load_iris()
X_iris, y_iris=iris.data, iris.target
print (X_iris.shape, y_iris.shape)
print (X_iris[0],y_iris[0])
#import the dataset and show its first instance
#(150, 4) (150,)
#[5.1 3.5 1.4 0.2] 0


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X, y=X_iris[:, :2], y_iris
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=33)
print (X_train.shape, y_train.shape)
#split the dataset into 2 sets
#(112, 2) (112,)


scaler=preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
#standardlize the features


import matplotlib.pyplot as plt
colors=['red','greenyellow','blue']
for i in range(len(colors)):
    xs=X_train[:,0][y_train==i]
    ys=X_train[:,1][y_train==i]
    plt.scatter(xs,ys,c=colors[i])
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
#display the distribution of trainning observations 


from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(X_train, y_train)
print( clf.coef_)
print( clf.intercept_)
#SGDClassifier's coefficients
#[[-22.7862496   10.65073386]
# [ -0.24123141  -3.91216097]
# [  8.6637987   -3.00995503]]
#[-11.30717666  -4.77300712  -3.38288123]


import numpy as np
x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
Xs = np.arange(x_min, x_max, 0.5)
fig, axes = plt.subplots(1, 3)
fig.set_size_inches(10, 6)
for i in [0, 1, 2]:
    axes[i].set_aspect('equal')
    axes[i].set_title('Class '+ str(i) + ' versus the rest')
    axes[i].set_xlabel('Sepal length')
    axes[i].set_ylabel('Sepal width')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    plt.sca(axes[i])
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.prism)
    ys = (-clf.intercept_[i] - Xs* clf.coef_[i, 0]) / clf.coef_[i, 1]
    plt.plot(Xs, ys)
#draw the classification boundaries


print (clf.predict(scaler.transform([[7.8,3.5]])))
#predict the class of flower with 7.8 length and 3.5 width sepal
#[2]


print( clf.decision_function(scaler.transform([[7.8, 3.5]])) )
#[[-54.82854951  -8.95925169  14.12716496]]


from sklearn import metrics
y_train_pred = clf.predict(X_train)
print( metrics.accuracy_score(y_train, y_train_pred) )
#the accuracy of prediction on the training set
#0.7678571428571429


y_pred = clf.predict(X_test)
print( metrics.accuracy_score(y_test, y_pred) )
#the accuracy of prediction on the test set
#0.7105263157894737


print( metrics.classification_report(y_test, y_pred, target_names=iris.target_names) )
#              precision    recall  f1-score   support
#      setosa       1.00      1.00      1.00         8
#  versicolor       0.50      0.09      0.15        11
#   virginica       0.64      0.95      0.77        19
#    accuracy                           0.71        38
#   macro avg       0.71      0.68      0.64        38
#weighted avg       0.68      0.71      0.64        38


print( metrics.confusion_matrix(y_test, y_pred) )
#the confusion matrix
#[[ 8  0  0]
# [ 0  1 10]
# [ 0  1 18]]

print("My name is {Yulong Wang}")
print("My NetID is: {yulongw2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


######Stop Here##########




##error in scikit learn package, which version??
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
# create a composite estimator made by a pipeline of the standarization and the linear model
clf = Pipeline([(
        'scaler', StandardScaler()),
        ('linear_model', SGDClassifier())
])
# create a k-fold cross validation iterator of k=5 folds
cv = KFold(5, shuffle=True, random_state=33)
# by default the score used is the one returned by score method of the estimator (accuracy)
scores = cross_val_score(clf, X, y, cv=cv)
print( scores )
#[0.73333333 0.73333333 0.8        0.86666667 0.83333333]


from scipy.stats import sem
def mean_score(scores): return ("Mean score: {0:.3f} (+/- {1:.3f})").format(np.mean(scores), sem(scores))
print( mean_score(scores) )
#Mean score: 0.793 (+/- 0.027)

