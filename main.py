# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd 
import numpy as np
import seaborn as sns
import math
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


tags = pd.read_csv('gender_submission.csv')
test_set = pd.read_csv('test.csv')
train_set = pd.read_csv('train.csv')

# Pre-processing: Analyze features to identify which ones might be useful in
# discerning survival rate
# First identify the number of NaN values in each feature
def compute_nan_stats(df):
    numberOfNaN = df.isnull().sum().sort_values(ascending = False)
    percentage = round((numberOfNaN/len(df))*100, 2)
    return pd.concat([numberOfNaN, percentage], axis = 1, 
                     keys = ['Number', 'Percentage'])
missing_train_data = compute_nan_stats(train_set)
missing_test_data = compute_nan_stats(test_set)
print(missing_train_data)
print(missing_test_data)

# Cabin is missing most of its data, so we will discard it. We have to deal 
# with the missing age and embarked values data ourselves. Since it is only 2 
# passengers, we can set the embarked value to the port which is the most 
# common. We can also fix the missing age issue by setting the age of each 
# missing passenger to the mean age of their passenger class. 
train_set['Embarked'].describe()
# The most common port is S, so we will set both missing values to S.
train_set['Embarked'] = train_set['Embarked'].fillna('S')

# There is also one nan Fare value for the test set. We will simply set this to
# the median value of the train set.
test_set.fillna(value = train_set['Fare'].median(), inplace = True)
# Now we form datasets with the identified features: Pclass, Sex, Age, FamSize, 
# Fare, and embark locations
train = train_set.drop(columns = ['PassengerId', 'Name', 'Ticket', 'SibSp', 
                                  'Parch', 'Cabin', 'Age'])
test = test_set.drop(columns = ['PassengerId', 'Name', 'Ticket', 'SibSp', 
                                  'Parch', 'Cabin', 'Age'])
    
# There are 177 missing age values. We can fill these missing values by setting 
# them with a random value drawn from a distribution based on the mean and 
# standard deviation of each Pclass' age
def fill_age(df):
    # We use the train_set statistics for both the train and test sets
    mean_age = train_set.groupby('Pclass')['Age'].mean()
    std_age = train_set.groupby('Pclass')['Age'].std()
    agelist = []
    for i, passenger in df.iterrows():
        if math.isnan(passenger['Age']):
            age = round(std_age[passenger['Pclass']] * np.random.randn() + 
                        mean_age[passenger['Pclass']],1)
        else:
            age = passenger['Age']
        agelist.append(age)
    return agelist

train['Age'] = fill_age(train_set)
test['Age'] = fill_age(test_set)

# Encode string data into numeric for gender and embark location features
train['Sex'] = train['Sex'].map({'female': 0, 'male': 1}) 
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}) 
test['Sex'] = test['Sex'].map({'female': 0, 'male': 1}) 
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}) 
# Fam_Size = SibSp(number of siblings+spouses) + Parch(number of parents + 
# children)
train['FamSize'] = train_set['SibSp'] + train_set['Parch']
test['FamSize'] = test_set['SibSp'] + test_set['Parch']

# The dataset is ready for classification algorithms
train.info()
test.info()
# Set the features and target variables for the train and test sets
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test = test
Y_test = tags.drop(columns = 'PassengerId')

accuracy = pd.Series() # Series to hold accuracy scores for various methods
# Method 1: Logistic Regression
model = LogisticRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
accuracy['Logistic Regression'] = accuracy_score(Y_test, predictions) * 100

# Method 2: K-Neighbors Classifier
# Build a model with k in range 1-20 and select the best k-value
def k_neighbors(X_train, Y_train, X_test, Y_test):
    best_k = 1
    for k in range(1,21):
        model = KNeighborsClassifier(n_neighbors = k)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        temp_acc = accuracy_score(Y_test, predictions) * 100
        if k != 1: 
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_k = k
        else:
            best_acc = temp_acc
    return (best_k, best_acc)    

k, acc = k_neighbors(X_train, Y_train, X_test, Y_test)
accuracy[f'{k}-Neighbors'] = acc
 
# Method 3: Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
accuracy['Gaussian Naive Bayes'] = accuracy_score(Y_test, predictions) * 100

# Method 4: Perceptron
model = Perceptron(max_iter=50, tol=None)
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
accuracy['Perceptron'] = accuracy_score(Y_test, predictions) * 100

# Method 5: Linear Support Vector Classifier
model = LinearSVC()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
accuracy['Linear SVC'] = accuracy_score(Y_test, predictions) * 100

# Method 6: Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
accuracy['Decision Tree'] = accuracy_score(Y_test, predictions) * 100

# Method 7: Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
accuracy['Random Forest'] = accuracy_score(Y_test, predictions) * 100







