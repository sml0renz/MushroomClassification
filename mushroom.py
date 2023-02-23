# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:28:57 2022

@author: Samantha Lorenz
"""

import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

mdt = pd.read_csv('C:/Users/18153/Downloads/mushroom.csv')

##encoding vars as 1 or 0 (same way as R)
#print( mdt["class"].value_counts() )
newClass = {"class": {"edible":1, "poisonous": 0} }
mdt = mdt.replace(newClass)

#print( mdt["bruises"].value_counts() )
mdt["bruises"] = mdt["bruises"]*1

#print( mdt["odor"].value_counts() )
newOdor = {"odor": {"none":1, "foul": 0, "spicy": 0, "fishy": 0, "anise": 0, "almond": 0, "pungent": 0, "creosote": 0, "musty": 0} }
mdt = mdt.replace(newOdor)

#print( mdt["ringType"].value_counts() )
newringType = {"ringType": {"pendant":1, "evanescent": 0, "large": 0, "flaring": 0, "none": 0} }
mdt = mdt.replace(newringType)

#print( mdt["population"].value_counts() )
newPopulation = {"population": {"several":1, "solitary": 0, "scattered": 0, "numerous": 0, "abundant": 0, "clustered": 0} }
mdt = mdt.replace(newPopulation) 

##training and testing

d  = {'class': mdt["class"], 'bruises': mdt["bruises"], 'odor': mdt["odor"], 'ringType': mdt["ringType"], 'population': mdt["population"]}
df = pd.DataFrame(data = d)
x = df[['bruises', 'odor', 'ringType', 'population']].copy()
x1 = np.array(x)
y = df[['class']]
y1 = np.array(y) 

x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2)

##naive bayes

from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

modBayes = BernoulliNB(alpha=0, fit_prior=True)
modBayes.fit(x_train, y_train )
prediction_prob = modBayes.predict_proba(x_test)
prediction = modBayes.predict(x_test)

print(confusion_matrix(y_test, prediction, labels=[0, 1]))

report = classification_report(y_test, prediction)
print(report)
accuracy = modBayes.score(x_test, y_test)
print(f'The accuracy is: {accuracy*100:.1f}%')

## SVM

from sklearn import svm
modSVM = svm.SVC(kernel='linear')
modSVM.fit(x_train, y_train)
svmPred = modSVM.predict(x_test)
print(confusion_matrix(y_test, svmPred, labels=[0, 1]))
svmReport = classification_report(y_test, svmPred)
print(svmReport)
accuracySVM = modSVM.score(x_test, y_test)
print(accuracySVM)

## logistic

from sklearn.linear_model import LogisticRegression 
modLog = LogisticRegression(penalty='l2')
modLog.fit(x_train, y_train)
logPred = modLog.predict(x_test)
print(confusion_matrix(y_test, logPred, labels=[0, 1]))
logReport = classification_report(y_test, logPred)
print(logReport)
accuracyLog = modLog.score(x_test, y_test)
print(accuracyLog)

## trees

from sklearn import tree
treeMod = tree.DecisionTreeClassifier()
treeMod.fit(x_train, y_train)
treePred = treeMod.predict(x_test)
print(confusion_matrix(y_test, treePred, labels=[0, 1]))
treeReport = classification_report(y_test, treePred)
print(treeReport)
accuracyTree = treeMod.score(x_test, y_test)
print(accuracyTree)

plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
tree.plot_tree(treeMod, fontsize=10)
plt.show()


## import graphviz 
## dot_data = tree.export_graphviz(treeMod, out_file=None) 
## graph = graphviz.Source(dot_data) 
## graph.render("Mushroom")

## dot_data = tree.export_graphviz(treeMod, out_file=None, 
##                      feature_names={"edible", "poisonous"},  
##                    class_names={"Bruises", "Odor", "Ring_type", "Population"},  
##                      filled=True, rounded=True,  
##                      special_characters=True)  
## graph = graphviz.Source(dot_data)  
## graph  