#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing our cancer dataset
dataset = pd.read_csv("/Users/bhargavmuppalla/Documents/Machine Learning/Project/cancer_data.csv")
df = pd.DataFrame(dataset)

X = df.loc[:, ~df.columns.isin(['diagnosis','Unnamed: 32'])]
Y = df.loc[:, df.columns == 'diagnosis']

#print(dataset)
#Encoding categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Using Logistic Regression Algorithm to the Training Set
from sklearn.linear_model import LogisticRegression
classifierlogistic = LogisticRegression(random_state = 0)
classifierlogistic.fit(X_train, Y_train)
Y_pred = classifierlogistic.predict(X_test)

from sklearn.metrics import confusion_matrix
cmLogistic = confusion_matrix(Y_test, Y_pred)
#print(cmLogistic)

#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
classifierknn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifierknn.fit(X_train, Y_train)
Y_pred = classifierlogistic.predict(X_test)

from sklearn.metrics import confusion_matrix
cmKnn = confusion_matrix(Y_test, Y_pred)
#print(cmKnn)

#Using SVC method of svm class to use Support Vector Machine Algorithm
from sklearn.svm import SVC
classifiersvm = SVC(kernel = 'linear', random_state = 0)
classifiersvm.fit(X_train, Y_train)
Y_pred = classifierlogistic.predict(X_test)

from sklearn.metrics import confusion_matrix
cmSvm = confusion_matrix(Y_test, Y_pred)
#print(cmSvm)

#Using SVC method of svm class to use Kernel SVM Algorithm
from sklearn.svm import SVC
classifiersvmk = SVC(kernel = 'rbf', random_state = 0)
classifiersvmk.fit(X_train, Y_train)

#Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
classifiergnb = GaussianNB()
classifiergnb.fit(X_train, Y_train)

#Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier
classifierdt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierdt.fit(X_train, Y_train)

#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm

from sklearn.ensemble import RandomForestClassifier
classifierrf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifierrf.fit(X_train, Y_train)

#predict
scoreLogistic = classifierlogistic.score(X_test,Y_test)
print("Accuracy of Logistic Regression Model: ", scoreLogistic)

scoreKNN = classifierknn.score(X_test,Y_test)
print("Accuracy of K nearest neighbour Model: ",scoreKNN)

scoreSVM = classifiersvm.score(X_test,Y_test)
print("Accuracy of Support Vector Machine Model: ",scoreSVM)

scoresvmk = classifiersvmk.score(X_test,Y_test)
print("Accuracy of Kernel SVM: ",scoresvmk)

scoregnb = classifiergnb.score(X_test,Y_test)
print("Accuracy of GaussianNB: ",scoregnb)

scoredt = classifierdt.score(X_test,Y_test)
print("Accuracy of DecisionTreeClassifier: ",scoredt)

scorerf= classifierrf.score(X_test,Y_test)
print("Accuracy of RandomForestClassifier: ",scorerf)
