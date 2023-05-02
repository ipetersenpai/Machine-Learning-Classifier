# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# Reading the heart disease dataset
heart_data = pd.read_csv("heart_disease.csv")

# Separating the features and target variable
X = heart_data.drop(columns=["target"])
y = heart_data["target"]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the Naïve Bayes classifier
nb_clf = GaussianNB()

# Training the classifier
nb_clf.fit(X_train, y_train)

#this will callculate the 5 fold cross validation

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
cv_results = cross_validate(nb_clf, X_train, y_train, cv=5, scoring=scoring)

# Predicting the target variable for the testing set
y_pred = nb_clf.predict(X_test)

# Calculating the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)

# Printing the accuracy of the classifier
print("Accuracy of Naïve Bayes classifier :", accuracy)

# Calculate precision, recall, specificity, sensitivity, and F1-score
report = classification_report(y_test, y_pred)
print(report)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
print("Specificity:", specificity)
print("Sensitivity:", sensitivity)

print("\nCross-validation scores:")
for key in scoring:
    print(key + ":", cv_results["test_" + key])
